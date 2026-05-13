from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from app.database import engine, SessionLocal, Base, get_db
from app.models import User, Chat, Conversation, Advocate
from app.schemas import UserCreate, AdvocateOut
from jose import jwt, JWTError
import json
import asyncio
from google import genai
import smtplib
import os
import random
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone


# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lexi-connect-frontend.vercel.app",        
                   "http://localhost:5173",
                    "http://localhost:3000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- JWT Configuration ---
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# --- Password hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# --- OTP Storage (in-memory) ---
otp_store = {}  # {email: {"otp": "123456", "expires": datetime}}


class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    otp: str

class ActivityItem(BaseModel):
    id: int
    type: str
    title: str
    time: str
    conversation_id: int

import threading
import requests

import os
def send_otp_email(email: str, otp: str):
    api_key = os.environ.get("BREVO_API_KEY")
    
    url = "https://api.brevo.com/v3/smtp/email"
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": api_key
    }
    
    payload = {
        "sender": {
            "name": "LexiConnect",
            "email": os.environ.get("EMAIL_SENDER")
        },
        "to": [{"email": email}],
        "subject": "LexiConnect - Your OTP Code",
        "htmlContent": f"""
            <h2>Your OTP Code</h2>
            <p>Your one-time password is: <strong>{otp}</strong></p>
            <p>This code expires in 10 minutes.</p>
            <p>If you didn't request this, ignore this email.</p>
        """
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 201:
        raise Exception(f"Brevo API error: {response.status_code} - {response.text}")

@app.post("/send-otp")
def send_otp_endpoint(request: OTPRequest):
    otp = str(random.randint(100000, 999999))
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    otp_store[request.email] = {"otp": otp, "expires": expires}
    
    def send_email_thread():
        try:
            send_otp_email(request.email, otp)
            print(f"✅ OTP sent to {request.email}")
        except Exception as e:
            print(f"❌ Email error: {e}")
    
    thread = threading.Thread(target=send_email_thread)
    thread.start()
    
    return {"message": "OTP sent successfully"}
    
    return {"message": "OTP sent successfully"}
# --- Verify OTP endpoint ---
@app.post("/verify-otp")
def verify_otp_endpoint(request: OTPVerify):
    record = otp_store.get(request.email)
    
    if not record:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
    
    if datetime.now(timezone.utc) > record["expires"]:
        del otp_store[request.email]
        raise HTTPException(status_code=400, detail="OTP has expired")
    
    if record["otp"] != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    del otp_store[request.email]
    return {"message": "OTP verified successfully"}

def hash_password(password: str):
    return pwd_context.hash(password)

def get_password_hash(password):
    password = password[:72]  # bcrypt limit
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    plain_password = plain_password[:72]  # bcrypt limit
    return pwd_context.verify(plain_password, hashed_password)

class ResetPasswordRequest(BaseModel):
    email: str
    new_password: str

# Tracks which emails have completed OTP verification (ready to reset)
verified_reset_emails = set()

@app.post("/verify-otp-reset")
def verify_otp_for_reset(request: OTPVerify):
    """Verify OTP specifically for password reset — marks email as verified."""
    record = otp_store.get(request.email)
    
    if not record:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
    
    if datetime.now(timezone.utc) > record["expires"]:
        del otp_store[request.email]
        raise HTTPException(status_code=400, detail="OTP has expired")
    
    if record["otp"] != request.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    del otp_store[request.email]
    verified_reset_emails.add(request.email)
    return {"message": "OTP verified successfully"}

@app.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password after OTP has been verified."""
    if request.email not in verified_reset_emails:
        raise HTTPException(status_code=403, detail="Email not verified. Please complete OTP verification first.")
    
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.password_hash = get_password_hash(request.new_password)
    db.commit()
    
    verified_reset_emails.discard(request.email)
    return {"message": "Password reset successfully"}


# --- JWT Token Functions ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Create tables ---
Base.metadata.create_all(bind=engine)

# --- Proper Authentication Dependency ---
def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    # Verify token and get user_id
    user_id = verify_token(token)
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user
def _relative_time(dt: datetime) -> str:
    """Convert a datetime to a human-readable relative string."""
    # Make sure both sides are offset-aware for comparison
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
 
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        m = seconds // 60
        return f"{m} minute{'s' if m > 1 else ''} ago"
    elif seconds < 86400:
        h = seconds // 3600
        return f"{h} hour{'s' if h > 1 else ''} ago"
    elif seconds < 604800:
        d = seconds // 86400
        return f"{d} day{'s' if d > 1 else ''} ago"
    else:
        w = seconds // 604800
        return f"{w} week{'s' if w > 1 else ''} ago"
 
 
@app.get("/activity/recent", response_model=List[ActivityItem])
def get_recent_activity(
    limit: int = 5,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Returns the most recent conversations for the logged-in user.
    Uses the conversation title (set automatically from the first message)
    as the activity label.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .all()
    )
 
    items = []
    for conv in conversations:
        title = conv.title or "Untitled conversation"
 
        # If the title is still the default, try to use the first user message
        if title in ("New Chat", "Untitled conversation"):
            first_msg = (
                db.query(Chat)
                .filter(Chat.conversation_id == conv.id, Chat.role == "user")
                .order_by(Chat.timestamp.asc())
                .first()
            )
            if first_msg and first_msg.content:
                title = first_msg.content
 
        # Truncate long titles
        if len(title) > 80:
            title = title[:77] + "..."
 
        items.append(ActivityItem(
            id=conv.id,
            type="chat",
            title=title,
            time=_relative_time(conv.updated_at),
            conversation_id=conv.id,
        ))
 
    return items
 
# --- Pydantic models ---
class UserOut(BaseModel):
    id: int
    name: str
    email: str
    phone: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    user: UserOut
    token: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    sources: List[str] = []
    status: str = "completed"
    message_id: int
    conversation_id: int

class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: Optional[str] = None

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None

# --- AI RAG system setup ---
rag = None
def generate_ai_response(user_input: str, conversation_history: list = None):
    global rag

    if rag is None:
        from app.rag.embeddings_groq import EnhancedLawRAGSystem
        rag = EnhancedLawRAGSystem.load_system("app/rag/law_rag_v3.pkl")
        rag.load_llm()
        print("📘 RAG System Ready!")

    # ✅ Reset RAG's internal history before every query
    rag._conversation_turns = []
    rag.memory = rag.__class__.__module__  # Reset structured memory too

    # ✅ Inject only THIS conversation's actual history
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                # We need pairs (user, assistant), so we'll build them below
                pass
    
    # Build proper turn pairs from conversation history
    if conversation_history:
        turns = []
        i = 0
        while i < len(conversation_history):
            if conversation_history[i]["role"] == "user":
                user_msg = conversation_history[i]["content"]
                # Look for the next assistant message
                assistant_msg = ""
                if i + 1 < len(conversation_history) and conversation_history[i+1]["role"] == "assistant":
                    assistant_msg = conversation_history[i+1]["content"] or ""
                    i += 2
                else:
                    i += 1
                if user_msg and assistant_msg:
                    turns.append((user_msg, assistant_msg))
            else:
                i += 1
        rag._conversation_turns = turns

    # Reset structured memory (fresh per conversation)
    from app.rag.embeddings_groq import StructuredMemory
    rag.memory = StructuredMemory()

    reply, sources = "", []
    if user_input.strip():
        result = rag.query(user_input)
        reply = result["answer"]
        sources = []

    return reply, sources

# --- Helper function to generate conversation title ---
def generate_conversation_title(first_message: str) -> str:
    """Generate a title from the first message (max 50 chars)"""
    title = first_message.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    return title

# --- Store for tracking message updates ---
message_updates = {}


@app.get("/lawyers", response_model=List[AdvocateOut])
def get_lawyers(
    city: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Advocate)

    # 🔍 Search filter
    if search:
        query = query.filter(
            (Advocate.name.ilike(f"%{search}%")) |
            (Advocate.primary_specialty.ilike(f"%{search}%")) |
            (Advocate.secondary_specialty.ilike(f"%{search}%"))
        )

    # 📍 City filter
    if city and city != "all":
        query = query.filter(Advocate.city == city)

    advocates = query.all()

    # 🔄 Transform DB → Frontend format
    result = []
    for adv in advocates:
        result.append({
            "id": str(adv.id),
            "name": adv.name,
            "rating": adv.rating or 0,
            "cases": adv.cases_handled or 0,
            "experience": adv.experience_years or 0,
            "specialization": list(filter(None, [
                adv.primary_specialty,
                adv.secondary_specialty
            ])),
            "city": adv.city,
            "phone": adv.phone_number,
            "email": adv.email,
            "address": adv.location
        })

    return result

def ensure_markdown_formatting(text: str) -> str:
    """If the text has no markdown formatting, reformat it."""
    has_markdown = any(c in text for c in ['**', '##', '- ', '* ', '`'])
    if has_markdown:
        return text  # Already formatted, skip
    
    lines = text.strip().split('\n')
    result = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect numbered list items like "1." "2." etc
        import re
        numbered = re.match(r'^(\d+)\.\s+(.+)', line)
        if numbered:
            result.append(f"{numbered.group(1)}. {numbered.group(2)}")
            continue
        
        # Detect lines ending with colon — treat as bold heading
        if line.endswith(':') and len(line) < 80:
            result.append(f"\n**{line}**")
            continue
        
        result.append(line)
    
    return '\n'.join(result)

# --- Background task to process AI response ---
def process_ai_response(message_id: int, user_input: str, conversation_id: int):
    db = SessionLocal()
    try:
        assistant_message = db.query(Chat).filter(Chat.id == message_id).first()
        if not assistant_message:
            return

        # ✅ Fetch this conversation's actual history (excluding the pending message)
        previous_messages = (
            db.query(Chat)
            .filter(
                Chat.conversation_id == conversation_id,
                Chat.id != message_id,
                Chat.status == "completed"
            )
            .order_by(Chat.timestamp.asc())
            .all()
        )

        # Format history the way most RAG/LLM systems expect
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in previous_messages
            if msg.content  # skip empty/pending messages
        ]

        reply, sources = generate_ai_response(user_input, conversation_history)
        reply = ensure_markdown_formatting(reply)

        assistant_message.content = reply
        assistant_message.sources = json.dumps(sources) if sources else None
        assistant_message.status = "completed"

        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.updated_at = datetime.now(timezone.utc)

        db.commit()

        message_updates[message_id] = {
            "status": "completed",
            "content": reply,
            "sources": sources,
            "timestamp": assistant_message.timestamp.isoformat()
        }

        print(f"✅ Completed response for message {message_id}")

    except Exception as e:
        print(f"❌ Error processing message {message_id}: {e}")
        assistant_message = db.query(Chat).filter(Chat.id == message_id).first()
        if assistant_message:
            assistant_message.status = "failed"
            assistant_message.content = "Error generating response"
            db.commit()
        message_updates[message_id] = {
            "status": "failed",
            "content": "Error generating response",
            "sources": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    finally:
        db.close()

# --- SSE endpoint for real-time updates ---
@app.get("/chat/stream/{message_id}")
async def stream_message_updates(
    message_id: int, 
    token: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Stream updates for a specific message using Server-Sent Events"""
    
    # Authenticate using token query parameter (EventSource doesn't support headers)
    if not token:
        async def error_generator():
            yield f"data: {json.dumps({'status': 'error', 'content': 'No token provided'})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )
    
    try:
        user_id = verify_token(token)
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            async def error_generator():
                yield f"data: {json.dumps({'status': 'error', 'content': 'Invalid token'})}\n\n"
            return StreamingResponse(
                error_generator(),
                media_type="text/event-stream"
            )
    except HTTPException:
        async def error_generator():
            yield f"data: {json.dumps({'status': 'error', 'content': 'Authentication failed'})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )
    
    async def event_generator():
        max_wait_time = 1200  # Maximum 5 minutes wait
        elapsed_time = 0
        check_interval = 0.5  # Check every 500ms
        
        while elapsed_time < max_wait_time:
            # Check if message has been updated in memory
            if message_id in message_updates:
                update = message_updates[message_id]
                yield f"data: {json.dumps(update)}\n\n"
                # Clean up
                del message_updates[message_id]
                break
            
            # Fallback: Check database status
            message = db.query(Chat).filter(
                Chat.id == message_id,
                Chat.user_id == user.id
            ).first()
            
            if not message:
                yield f"data: {json.dumps({'status': 'error', 'content': 'Message not found'})}\n\n"
                break
                
            if message.status != "pending":
                update = {
                    "status": message.status,
                    "content": message.content,
                    "sources": json.loads(message.sources) if message.sources else [],
                    "timestamp": message.timestamp.isoformat()
                }
                yield f"data: {json.dumps(update)}\n\n"
                break
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Timeout
        if elapsed_time >= max_wait_time:
            yield f"data: {json.dumps({'status': 'timeout', 'content': 'Request timed out'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

# --- Conversation endpoints ---
@app.post("/conversations", response_model=ConversationOut)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=user.id,
        title=request.title or "New Chat"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "message_count": 0,
        "last_message": None
    }

@app.get("/conversations")
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Get all conversations for the current user"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    result = []
    for conv in conversations:
        # Get message count
        message_count = db.query(Chat).filter(Chat.conversation_id == conv.id).count()
        
        # Get last message
        last_message = db.query(Chat).filter(
            Chat.conversation_id == conv.id
        ).order_by(Chat.timestamp.desc()).first()
        
        result.append({
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": message_count,
            "last_message": last_message.content[:100] if last_message and last_message.content else None
        })
    
    return result

@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get a specific conversation with its messages"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get all messages in this conversation
    messages = db.query(Chat).filter(
        Chat.conversation_id == conversation_id
    ).order_by(Chat.timestamp.asc()).all()
    
    messages_list = []
    for msg in messages:
        messages_list.append({
            "id": msg.id,
            "type": msg.role,
            "content": msg.content,
            "sources": json.loads(msg.sources) if msg.sources else [],
            "timestamp": msg.timestamp.isoformat(),
            "status": msg.status
        })
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "messages": messages_list
    }

@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}

@app.patch("/conversations/{conversation_id}/title")
def update_conversation_title(
    conversation_id: int,
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Update conversation title"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if request.title:
        conversation.title = request.title
        db.commit()
    
    return {"message": "Title updated successfully"}

# --- Chat endpoints ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db), 
    user: User = Depends(get_current_user)
):
    """Send a message in a conversation"""
    # Get or create conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == user.id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation with title from first message
        conversation = Conversation(
            user_id=user.id,
            title=generate_conversation_title(request.message)
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    # Save user message
    user_message = Chat(
        conversation_id=conversation.id,
        user_id=user.id,
        role="user",
        content=request.message,
        status="completed"
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Create pending assistant message
    assistant_message = Chat(
        conversation_id=conversation.id,
        user_id=user.id,
        role="assistant",
        content=None,  # Will be filled by background task
        status="pending"
    )
    db.add(assistant_message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(assistant_message)

    # Process AI response in background
    background_tasks.add_task(
        process_ai_response, 
        assistant_message.id, 
        request.message,
        conversation.id
    )

    return {
        "reply": None,
        "sources": [],
        "status": "pending",
        "message_id": assistant_message.id,
        "conversation_id": conversation.id
    }

@app.get("/chat/history")
def chat_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Get all messages for the current user across all conversations (deprecated - use /conversations instead)"""
    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.timestamp.asc()).all()
    result = []
    for chat in chats:
        result.append({
            "id": chat.id,
            "conversation_id": chat.conversation_id,
            "type": chat.role,
            "content": chat.content,
            "sources": json.loads(chat.sources) if chat.sources else [],
            "timestamp": chat.timestamp.isoformat(),
            "status": chat.status
        })
    return result

@app.get("/chat/message/{message_id}")
def get_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Check status of a specific message"""
    message = db.query(Chat).filter(
        Chat.id == message_id,
        Chat.user_id == user.id
    ).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "type": message.role,
        "content": message.content,
        "sources": json.loads(message.sources) if message.sources else [],
        "timestamp": message.timestamp.isoformat(),
        "status": message.status
    }

# --- Sign-up endpoint ---
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = User(
        name=user.name,
        email=user.email,
        phone=user.phone,
        password_hash=hash_password(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User created successfully"}

# --- Login endpoint ---
@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create JWT token
    token = create_access_token({"user_id": user.id})
    
    user_out = UserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        phone=user.phone
    )
    
    return {
        "user": user_out,
        "token": token
    }

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)