import uuid
from datetime import datetime, timezone
import numpy as np
from extensions import db


class FaceIdentity(db.Model):
    """
    A named identity — one row per known person.
    Can have many FaceEmbeddings (multiple enrollment images).
    """
    __tablename__ = "face_identities"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(120), unique=True, nullable=False, index=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    created_by = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=True)

    embeddings = db.relationship(
        "FaceEmbedding", backref="identity", lazy="dynamic", cascade="all, delete-orphan"
    )
    violations = db.relationship("Violation", backref="identity", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "embedding_count": self.embeddings.count(),
        }


class FaceEmbedding(db.Model):
    """
    A single 128-dimensional face embedding for a FaceIdentity.
    Multiple embeddings per identity dramatically improve match robustness.
    """
    __tablename__ = "face_embeddings"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"), nullable=False, index=True
    )
    # 128 float32 values stored as raw bytes — 512 bytes per row, no JSON overhead
    embedding_bytes = db.Column(db.LargeBinary(512), nullable=False)
    source_image = db.Column(db.Text, nullable=True)   # relative path, for reference only
    quality_score = db.Column(db.Float, nullable=False, default=1.0)
    # 1.0 = excellent clean enrollment, 0.0 = worst possible (never stored below threshold)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    @property
    def embedding(self) -> np.ndarray:
        """Deserialise bytes → float32 numpy array."""
        return np.frombuffer(self.embedding_bytes, dtype=np.float32).copy()

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        """Serialise float32 numpy array → bytes."""
        self.embedding_bytes = value.astype(np.float32).tobytes()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "identity_id": self.identity_id,
            "quality_score": round(self.quality_score, 4),
            "created_at": self.created_at.isoformat(),
        }


class Violation(db.Model):
    """
    A single PPE violation event.
    identity_id is nullable — unknown violators are stored without a link
    and can be retrospectively assigned via /face/merge.
    """
    __tablename__ = "violations"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    violation_type = db.Column(db.String(60), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=True)

    # Identity link — nullable until face is matched or manually assigned
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"), nullable=True, index=True
    )
    raw_name = db.Column(db.String(120), nullable=False, default="Unknown Person")
    match_distance = db.Column(db.Float, nullable=True)
    # The Euclidean face distance at match time — stored so operators can review
    # borderline matches.  NULL when no face was found in the crop.

    stream_id = db.Column(db.String(36), nullable=True, index=True)
    image_filename = db.Column(db.Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "violation_type": self.violation_type,
            "confidence": self.confidence,
            "name": self.identity.name if self.identity else self.raw_name,
            "identity_id": self.identity_id,
            "match_distance": self.match_distance,
            "stream_id": self.stream_id,
            "image_path": f"/api/v1/violations/image/{self.image_filename}"
            if self.image_filename
            else None,
        }
