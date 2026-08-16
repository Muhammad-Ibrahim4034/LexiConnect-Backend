# LexiConnect Backend

Backend API for **LexiConnect**, an AI-powered legal assistance platform that combines a Retrieval-Augmented Generation (RAG) system with user authentication, persistent conversations, and advocate search.

The backend is built with **FastAPI** and provides APIs for interacting with the legal AI assistant, managing conversations, and finding legal advocates.

## Features

* AI-powered legal question answering using RAG
* Legal document retrieval with vector search
* Conversation history and persistent chat storage
* JWT-based user authentication
* Streaming AI responses using Server-Sent Events
* Advocate search by name, specialty, and city
* PostgreSQL/SQLite database support
* Structured API responses using Pydantic
* Docker support
* Integration with LLM APIs through Groq and Google GenAI

## Architecture

```text
Client Application
       |
       v
   FastAPI API
       |
   +---+-------------------+
   |                       |
   v                       v
Authentication         Database
   |                 Users / Chats /
   |                 Conversations /
   |                 Advocates
   |
   v
Legal RAG System
       |
       +--> Sentence Transformers
       |
       +--> ChromaDB
       |
       +--> Legal Documents
       |
       +--> LLM
       |
       v
Legal AI Response
```

## RAG Pipeline

The legal assistant uses a Retrieval-Augmented Generation pipeline to provide answers based on a collection of legal documents.

```text
User Question
      ↓
Text Embedding
      ↓
Vector Search
      ↓
Relevant Legal Documents
      ↓
Context Construction
      ↓
LLM
      ↓
Generated Legal Response
```

The RAG implementation is located under `app/rag/` and includes the trained RAG system and embedding/retrieval logic.

## Tech Stack

* Python
* FastAPI
* SQLAlchemy
* Pydantic
* JWT Authentication
* Sentence Transformers
* ChromaDB
* PyTorch
* Hugging Face Transformers
* Groq
* Google GenAI
* PostgreSQL / SQLite
* Docker

The project's dependencies include FastAPI, SQLAlchemy, JWT libraries, Sentence Transformers, ChromaDB, PyTorch, Transformers, Groq, and Google GenAI.

## Project Structure

```text
LexiConnect-Backend/
│
├── app/
│   ├── data/
│   │   └── metadata/
│   │
│   ├── rag/
│   │   ├── embeddings_groq.py
│   │   └── law_rag_v3.pkl
│   │
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── server.py
│
├── chroma_store/
├── Dockerfile
├── requirements.txt
├── users.db
└── README.md
```

The main application logic is implemented in `app/server.py`, while database models cover users, conversations, chat messages, and advocates.

## API Capabilities

### Authentication

* User registration
* User login
* JWT token authentication
* Protected API endpoints

### Chat

* Send legal questions
* Retrieve AI-generated responses
* Store user and assistant messages
* Maintain conversation history
* Stream responses using Server-Sent Events

### Conversations

* Create conversations
* List user conversations
* Retrieve conversation history
* Persist messages between sessions

### Advocate Search

* Search advocates by name
* Filter by city
* Filter by legal specialization
* Retrieve advocate information

The backend exposes dedicated conversation endpoints and stores conversations and messages in the database.

## Installation

Clone the repository:

```bash
git clone https://github.com/Muhammad-Ibrahim4034/LexiConnect-Backend.git
cd LexiConnect-Backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and configure the required database, authentication, and LLM API settings.

## Running the Server

Start the FastAPI application with:

```bash
uvicorn app.server:app --reload
```

The API can then be accessed locally through:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Database

SQLAlchemy is used as the ORM. The current models include:

* `User`
* `Conversation`
* `Chat`
* `Advocate`

Chat messages are associated with both users and conversations, allowing conversation history to persist across sessions.

## Docker

The repository also includes a `Dockerfile` for containerized deployment.

Build and run the container:

```bash
docker build -t lexiconnect-backend .
docker run -p 8000:8000 lexiconnect-backend
```

## Future Improvements

* Improve legal document citation and source display
* Add more legal datasets and jurisdictions
* Improve RAG retrieval and reranking
* Add automated API testing
* Add production database migrations
* Improve deployment and monitoring
* Add role-based access control

## Author

**Muhammad Ibrahim**
BS Artificial Intelligence — FAST NUCES

---
