import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, Text
from sqlalchemy.dialects.postgresql import JSONB

from extensions import db


def _new_id(): return str(uuid.uuid4())
def _now():    return datetime.now(timezone.utc)

_label_lock = threading.Lock()


class FaceIdentity(db.Model):
    __tablename__  = "face_identities"
    __table_args__ = (
        Index("ix_fi_is_confirmed",        "is_confirmed"),
        Index("ix_fi_is_archived",         "is_archived"),
        Index("ix_fi_last_seen",           "last_seen"),
        Index("ix_fi_created_at",          "created_at"),
        Index("ix_fi_active_unconfirmed",  "is_archived", "is_confirmed", "last_seen"),
        {"extend_existing": True},
    )

    id             = db.Column(db.String(36), primary_key=True, default=_new_id)
    label          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    is_confirmed   = db.Column(db.Boolean, nullable=False, default=False)
    is_archived    = db.Column(db.Boolean, nullable=False, default=False)
    merged_into_id = db.Column(db.String(36), db.ForeignKey("face_identities.id"), nullable=True)

    created_at  = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    updated_at  = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=_now,
        onupdate=_now,
    )
    last_seen   = db.Column(db.DateTime(timezone=True), nullable=True)

    thumbnail_filename  = db.Column(db.Text, nullable=True)
    identity_confidence = db.Column(db.Float, nullable=False, default=0.0)

    meta = db.Column(JSONB, nullable=True, default=None)

    embeddings = db.relationship(
        "FaceEmbedding", backref="identity", lazy="dynamic",
        cascade="all, delete-orphan",
        foreign_keys="FaceEmbedding.identity_id",
    )
    violations = db.relationship(
        "Violation", backref="identity", lazy="dynamic",
        foreign_keys="Violation.identity_id",
    )
    stream_events = db.relationship(
        "StreamEvent", backref="identity", lazy="dynamic",
        foreign_keys="StreamEvent.identity_id",
    )

    @staticmethod
    def next_label() -> str:
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
            "updated_at":          self.updated_at.isoformat() if self.updated_at else None,
            "last_seen":           self.last_seen.isoformat() if self.last_seen else None,
            "embedding_count":     self.embeddings.count(),
            "violation_count":     self.violations.count(),
            "identity_confidence": round(self.identity_confidence or 0.0, 3),
            "thumbnail":           thumbnail,
            "meta":                self.meta,
        }

    def to_summary(self) -> dict:
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
    __table_args__ = (
        Index("ix_fe_identity_created", "identity_id", "created_at"),
        Index("ix_fe_identity_quality", "identity_id", "quality_score"),
        Index("ix_fe_stream_id",        "stream_id"),
        {"extend_existing": True},
    )

    id          = db.Column(db.String(36), primary_key=True, default=_new_id)
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True,
    )

    embedding_vec = db.Column(Vector(512), nullable=False)

    det_score     = db.Column(db.Float,      nullable=True)
    quality_score = db.Column(db.Float,      nullable=True)
    stream_id     = db.Column(db.String(36), nullable=True)
    created_at    = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)

    @property
    def embedding(self) -> np.ndarray:
        # pgvector may return a Python list or a float64 array; force float32.
        arr = np.asarray(self.embedding_vec, dtype=np.float32).ravel()
        # Re-normalise: pgvector float32 round-trip introduces norm drift that
        # causes dot products to silently fall below true cosine similarity.
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        arr  = np.asarray(value, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        self.embedding_vec = (arr / norm if norm > 0 else arr).tolist()


class Violation(db.Model):
    __tablename__  = "violations"
    __table_args__ = (
        Index("ix_viol_identity_timestamp",  "identity_id",    "timestamp"),
        Index("ix_viol_type_timestamp",      "violation_type", "timestamp"),
        Index("ix_viol_stream_timestamp",    "stream_id",      "timestamp"),
        {"extend_existing": True},
    )

    id             = db.Column(db.String(36), primary_key=True, default=_new_id)
    timestamp      = db.Column(db.DateTime(timezone=True), nullable=False, default=_now)
    violation_type = db.Column(db.String(60), nullable=False)
    confidence     = db.Column(db.Float,  nullable=True)
    identity_id    = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True,
    )
    match_score    = db.Column(db.Float,      nullable=True)
    stream_id      = db.Column(db.String(36), nullable=True)
    image_filename = db.Column(db.Text,       nullable=True)

    meta = db.Column(JSONB, nullable=True, default=None)

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
            "meta":           self.meta,
        }


class StreamEvent(db.Model):
    __tablename__  = "stream_events"
    __table_args__ = (
        Index("ix_se_stream_timestamp",   "stream_id",   "timestamp"),
        Index("ix_se_identity_timestamp", "identity_id", "timestamp"),
        Index("ix_se_event_type",         "event_type"),
        Index("ix_se_track_id",           "track_id"),
        {"extend_existing": True},
    )

    EVT_TRACK_CONFIRMED   = "track_confirmed"
    EVT_IDENTITY_MATCHED  = "identity_matched"
    EVT_IDENTITY_CREATED  = "identity_created"
    EVT_VIOLATION         = "violation"
    EVT_TRACK_LOST        = "track_lost"
    EVT_STREAM_STARTED    = "stream_started"
    EVT_STREAM_STOPPED    = "stream_stopped"

    id          = db.Column(db.String(36), primary_key=True, default=_new_id)
    stream_id   = db.Column(db.String(36), nullable=False)
    track_id    = db.Column(db.Integer,    nullable=True)
    identity_id = db.Column(
        db.String(36), db.ForeignKey("face_identities.id"),
        nullable=True,
    )
    event_type  = db.Column(db.String(60), nullable=False)
    timestamp   = db.Column(db.DateTime(timezone=True), nullable=False, default=_now, index=True)
    confidence  = db.Column(db.Float, nullable=True)

    meta = db.Column(JSONB, nullable=True, default=None)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "stream_id":   self.stream_id,
            "track_id":    self.track_id,
            "identity_id": self.identity_id,
            "event_type":  self.event_type,
            "timestamp":   self.timestamp.isoformat(),
            "confidence":  self.confidence,
            "meta":        self.meta,
        }

    def __repr__(self) -> str:
        return f"<StreamEvent {self.event_type!r} stream={self.stream_id[:8]} t={self.timestamp}>"