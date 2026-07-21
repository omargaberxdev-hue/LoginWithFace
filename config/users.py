from sqlalchemy import String, Index
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from .base import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    age: Mapped[str] = mapped_column(String(50))
    embedding: Mapped[list[float]] = mapped_column(Vector(512))  # must match ImageEmbedding's actual output

    __table_args__ = (
        Index(
            "ix_users_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )