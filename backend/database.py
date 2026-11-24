import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, event
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "driftguard.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InferenceLog(Base):
    __tablename__ = "inference_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    amount = Column(Float)
    distance_from_home = Column(Float)
    hour_of_day = Column(Integer)
    tx_frequency_24h = Column(Integer)
    prediction = Column(Integer)
    fraud_probability = Column(Float)

Base.metadata.create_all(bind=engine)