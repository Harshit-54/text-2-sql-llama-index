from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Identity, Column, Integer, String

class Base(DeclarativeBase):
    pass

class TestTable(Base):
    __tablename__ = "test_table"

    id_identity = Identity(start = 1, increment=1)
    id= Column(Integer, id_identity, primary_key=True)

    test_column1 = Column(String(50), unique=True, nullable=True)