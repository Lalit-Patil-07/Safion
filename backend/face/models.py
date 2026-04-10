"""
Face + Violation models — v1.1
==============================
FaceIdentity  — persistent identity record (never hard-deleted)
FaceEmbedding — 512-dim ArcFace embedding; many per identity
Violation     — PPE event; always linked to an identity
"""
import uuid
from datetime import datetime, timezone

import numpy as np
from extensions import db


def _new_id(): return str(uuid.uuid4())
def _now():    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
class FaceIdentity(db.Model):
    __tablename__ = "face_identities"
    __table_args__ = {"extend_existing": True}

    id           = db.Column(db.String(36), primary_key=True, default=_new_id)
    label        = db.Column(db.String(120), unique=True, nullable=False, index=True)
    is_confirmed = db.Column(db.Boolean, nullable=False, default=False)
    is_archived  = db.Column(db.Boolean, nullable=False, default=False, index=True)
    # archived = merged into another identity; invisible in normal queries
    merged_into_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"), nullable=True
    )

    created_at  = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    last_seen   = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    # ^ updated whenever a new violation is logged for this identity

    # Stable face thumbnail — set once when first good violation is saved
    thumbnail_filename = db.Column(db.Text, nullable=True)

    embeddings = db.relationship(
        "FaceEmbedding", backref="identity", lazy="dynamic",
        cascade="all, delete-orphan",
        foreign_keys="FaceEmbedding.identity_id",
    )
    violations = db.relationship(
        "Violation", backref="identity", lazy="dynamic",
        foreign_keys="Violation.identity_id",
    )

    # ── Label generation ──────────────────────────────────────────────────────
    @staticmethod
    def next_label() -> str:
        """
        Collision-safe sequential label.
        Reads the highest existing numeric suffix and increments it.
        Thread-safe because DB insert will fail on unique constraint
        and the caller can retry.
        """
        from sqlalchemy import func, text
        # Extract numeric part from labels like "Person_042"
        result = db.session.execute(
            text(
                "SELECT COALESCE(MAX(CAST(SUBSTR(label, 8) AS INTEGER)), 0) "
                "FROM face_identities WHERE label LIKE 'Person_%'"
            )
        ).scalar()
        return f"Person_{int(result) + 1:03d}"

    # ── Centroid (score-weighted) ──────────────────────────────────────────────
    def centroid(self) -> np.ndarray | None:
        """
        Weighted mean embedding, using det_score as weight.
        Higher-confidence detections influence the centroid more.
        Falls back to equal weighting when det_score is unavailable.
        """
        rows = self.embeddings.all()
        if not rows:
            return None
        embs    = np.stack([r.embedding for r in rows])
        weights = np.array([r.det_score if r.det_score else 0.7 for r in rows])
        weights = weights / weights.sum()
        mean    = (embs * weights[:, None]).sum(axis=0).astype(np.float32)
        norm    = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """
        Uses pre-stored thumbnail_filename — no extra query per serialisation.
        last_seen is stored on the row — no extra query.
        """
        thumbnail = (
            f"/violations/image/{self.thumbnail_filename}"
            if self.thumbnail_filename else None
        )
        return {
            "id":              self.id,
            "label":           self.label,
            "is_confirmed":    self.is_confirmed,
            "is_archived":     self.is_archived,
            "created_at":      self.created_at.isoformat(),
            "last_seen":       self.last_seen.isoformat() if self.last_seen else None,
            "embedding_count": self.embeddings.count(),
            "violation_count": self.violations.count(),
            "thumbnail":       thumbnail,
        }

    def to_summary(self) -> dict:
        """
        Lightweight version for list views — avoids .count() calls.
        Call this when rendering identity cards.
        """
        thumbnail = (
            f"/violations/image/{self.thumbnail_filename}"
            if self.thumbnail_filename else None
        )
        return {
            "id":           self.id,
            "label":        self.label,
            "is_confirmed": self.is_confirmed,
            "is_archived":  self.is_archived,
            "last_seen":    self.last_seen.isoformat() if self.last_seen else None,
            "thumbnail":    thumbnail,
        }


# ──────────────────────────────────────────────────────────────────────────────
class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"
    __table_args__ = {"extend_existing": True}

    id          = db.Column(db.String(36), primary_key=True, default=_new_id)
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True, index=True,
    )
    embedding_bytes = db.Column(db.LargeBinary(2048), nullable=False)
    det_score   = db.Column(db.Float, nullable=True)
    stream_id   = db.Column(db.String(36), nullable=True, index=True)
    created_at  = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)

    @property
    def embedding(self) -> np.ndarray:
        return np.frombuffer(self.embedding_bytes, dtype=np.float32).copy()

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        arr  = np.asarray(value, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        self.embedding_bytes = (arr / norm if norm > 0 else arr).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
class Violation(db.Model):
    __tablename__ = "violations"
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