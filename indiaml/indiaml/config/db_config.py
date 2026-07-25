import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..models.models import Base

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATABASE_URL = os.environ.get("INDIAML_DATABASE_URL", "sqlite:///venues.db")

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

def init_db():
    """Create all tables. Call this once at startup."""
    Base.metadata.create_all(bind=engine)
