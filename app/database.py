from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time

# SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

# Create engine with proper SQLite configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0  # 30 second timeout for locked database
    },
    echo=False
)

# Enable WAL mode and other pragmas for better concurrency
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    
    # Try to set WAL mode with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Other optimizations
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            
            cursor.close()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Attempt {attempt + 1} to configure database failed, retrying...")
                time.sleep(1)
                cursor = dbapi_conn.cursor()
            else:
                print(f"⚠️  Warning: Could not enable WAL mode: {e}")
                print("    Database will work but with reduced concurrency")
                try:
                    cursor.close()
                except:
                    pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()