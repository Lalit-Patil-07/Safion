"""
Face + Violation models — v1.3
Embedding storage migrated from LargeBinary to pgvector Vector(512).
External interface (embedding property) is unchanged.
"""
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from pgvector.sqlalchemy import Vector
from extensions import db


def _new_id(): return str(uuid.uuid4())
def _now():    return datetime.now(timezone.utc)

# Serialises next_label() + flush() so no two threads can interleave them
# and produce duplicate auto-labels.
_label_lock = threading.Lock()


class FaceIdentity(db.Model):
    __tablename__  = "face_identities"
    __table_args__ = {"extend_existing": True}

    id             = db.Column(db.String(36), primary_key=True, default=_new_id)
    label          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    is_confirmed   = db.Column(db.Boolean, nullable=False, default=False)
    is_archived    = db.Column(db.Boolean, nullable=False, default=False, index=True)
    merged_into_id = db.Column(db.String(36), db.ForeignKey("face_identities.id"), nullable=True)

    created_at  = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    last_seen   = db.Column(db.DateTime(timezone=True), nullable=True, index=True)

    thumbnail_filename  = db.Column(db.Text, nullable=True)
    identity_confidence = db.Column(db.Float, nullable=False, default=0.0)

    embeddings = db.relationship(
        "FaceEmbedding", backref="identity", lazy="dynamic",
        cascade="all, delete-orphan",
        foreign_keys="FaceEmbedding.identity_id",
    )
    violations = db.relationship(
        "Violation", backref="identity", lazy="dynamic",
        foreign_keys="Violation.identity_id",
    )

    @staticmethod
    def next_label() -> str:
        """Collision-safe sequential label using MAX(numeric suffix)."""
        from sqlalchemy import text
        result = db.session.execute(
            text(
                "SELECT COALESCE(MAX(CAST(SUBSTR(label, 8) AS INTEGER)), 0) "
                "FROM face_identities WHERE label LIKE 'Person_%'"
            )
        ).scalar()
        return f"Person_{int(result) + 1:03d}"

    @classmethod
    def create_auto(cls) -> "FaceIdentity":
        """
        Atomically generate a unique auto-label and flush the new identity.
        Holds _label_lock so no two threads produce the same label.
        """
        with _label_lock:
            label    = cls.next_label()
            identity = cls(label=label, is_confirmed=False, identity_confidence=0.0)
            db.session.add(identity)
            db.session.flush()
            return identity

    def to_dict(self) -> dict:
        thumbnail = (
            f"/violations/image/{self.thumbnail_filename}"
            if self.thumbnail_filename else None
        )
        return {
            "id":                  self.id,
            "label":               self.label,
            "is_confirmed":        self.is_confirmed,
            "is_archived":         self.is_archived,
            "created_at":          self.created_at.isoformat(),
            "last_seen":           self.last_seen.isoformat() if self.last_seen else None,
            "embedding_count":     self.embeddings.count(),
            "violation_count":     self.violations.count(),
            "identity_confidence": round(self.identity_confidence or 0.0, 3),
            "thumbnail":           thumbnail,
        }

    def to_summary(self) -> dict:
        """Lightweight — uses stored columns only, no .count() subqueries."""
        thumbnail = (
            f"/violations/image/{self.thumbnail_filename}"
            if self.thumbnail_filename else None
        )
        return {
            "id":                  self.id,
            "label":               self.label,
            "is_confirmed":        self.is_confirmed,
            "is_archived":         self.is_archived,
            "last_seen":           self.last_seen.isoformat() if self.last_seen else None,
            "identity_confidence": round(self.identity_confidence or 0.0, 3),
            "thumbnail":           thumbnail,
        }


class FaceEmbedding(db.Model):
    __tablename__  = "face_embeddings"
    __table_args__ = {"extend_existing": True}

    id          = db.Column(db.String(36), primary_key=True, default=_new_id)
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True, index=True,
    )

    # pgvector column — stores 512-dim L2-normalised ArcFace embedding.
    # Replaces the previous LargeBinary(2048) bytes column.
    # The `embedding` property below keeps the external interface identical
    # (accepts/returns np.ndarray) so pipeline, clustering, and similarity
    # code requires no changes.
    embedding_vec = db.Column(Vector(512), nullable=False)

    det_score     = db.Column(db.Float, nullable=True)
    quality_score = db.Column(db.Float, nullable=True)
    stream_id     = db.Column(db.String(36), nullable=True, index=True)
    created_at    = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)

    @property
    def embedding(self) -> np.ndarray:
        """Return the stored vector as a float32 numpy array."""
        return np.asarray(self.embedding_vec, dtype=np.float32)

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        """Accept a numpy array, L2-normalise, and store as a plain Python list."""
        arr  = np.asarray(value, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        normalised = arr / norm if norm > 0 else arr
        self.embedding_vec = normalised.tolist()


class Violation(db.Model):
    __tablename__  = "violations"
    __table_args__ = {"extend_existing": True}

    id             = db.Column(db.String(36), primary_key=True, default=_new_id)
    timestamp      = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)
    violation_type = db.Column(db.String(60), nullable=False, index=True)
    confidence     = db.Column(db.Float, nullable=True)
    identity_id    = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True, index=True,
    )
    match_score    = db.Column(db.Float, nullable=True)
    stream_id      = db.Column(db.String(36), nullable=True, index=True)
    image_filename = db.Column(db.Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "timestamp":      self.timestamp.isoformat(),
            "violation_type": self.violation_type,
            "confidence":     self.confidence,
            "name":           self.identity.label if self.identity else "Unknown Person",
            "identity_id":    self.identity_id,
            "match_score":    round(self.match_score, 4) if self.match_score else None,
            "stream_id":      self.stream_id,
            "image_path":     f"/violations/image/{self.image_filename}"
                              if self.image_filename else None,
        }