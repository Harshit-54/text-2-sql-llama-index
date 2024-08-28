from app.settings  import get_oracledb_settings
from sqlalchemy import (
    create_engine,
    MetaData)
from app import database_models

from sqlalchemy.orm import Session

oracledb_settings = get_oracledb_settings()


username = oracledb_settings.ORACLE_DB_USERNAME
password = oracledb_settings.ORACLE_DB_PASSWORD
dsn = oracledb_settings.ORACLE_DB_DSN

engine_oracledb = create_engine(f"oracle+oracledb://{username}:{password}@{dsn}")
database_models.Base.metadata.create_all(bind=engine_oracledb)

def get_db():
    # init_oracledb()
    db = Session(engine_oracledb)
    try:
        yield db
    finally:
        db.close()