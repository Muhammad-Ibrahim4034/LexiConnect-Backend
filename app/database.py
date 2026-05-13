from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get DATABASE_URL from Railway environment
# Falls back to SQLite only for local development
SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./users.db")

# Railway provides 'postgres://' but SQLAlchemy needs 'postgresql://'
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# PostgreSQL engine (no SQLite-specific args needed)
if SQLALCHEMY_DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_size=5,          # Max 5 persistent connections
        max_overflow=10,      # Up to 10 extra connections under load
        pool_pre_ping=True,   # Test connection before using (handles dropped connections)
        echo=False
    )
else:
    # Local SQLite fallback (your existing config)
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 30.0
        },
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()