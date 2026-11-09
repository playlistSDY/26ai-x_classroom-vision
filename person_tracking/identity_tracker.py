"""Associate temporary tracker IDs with session-long person identities."""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from face_identity import FaceGallery


@dataclass
class TrackerConfig:
    match_threshold: float = 0.68
    ambiguity_margin: float = 0.06
    continuity_seconds: float = 1.0
    continuity_min_similarity: float = 0.45
    confirmation_hits: int = 2
    new_identity_seconds: float = 1.0
    new_identity_hits: int = 3
    min_quality: float = 0.45
    min_confidence: float = 0.30
    gallery_size: int = 24
    gallery_interval: float = 0.35
    gallery_similarity: float = 0.98
    view_confirmation_hits: int = 3
    view_confirmation_seconds: float = 0.4
    view_consistency_threshold: float = 0.80
    view_min_quality: float = 0.7
    candidate_memory_seconds: float = 1.0
    part_match_threshold: float = 0.80
    part_ambiguity_margin: float = 0.09
    part_min_visibility: float = 0.5
    part_min_common: int = 2
    part_confirmation_hits: int = 3
    part_gallery_size: int = 24
    part_groups: tuple[int, ...] = (0, 1, 1, 2, 3, 3, 4, 4)
    face_min_quality: float = 0.7
    face_match_threshold: float = 0.50
    face_ambiguity_margin: float = 0.08
    face_new_threshold: float = 0.30
    face_confirmation_hits: int = 2
    face_gallery_size: int = 12

    def __post_init__(self):
        if (self.gallery_size < 2 or self.confirmation_hits < 1
                or self.new_identity_hits < 1 or self.view_confirmation_hits < 2):
            raise ValueError("Gallery size must be >= 2 and confirmation counts must be positive.")
        if min(self.continuity_seconds, self.new_identity_seconds,
               self.gallery_interval, self.view_confirmation_seconds,
               self.candidate_memory_seconds) < 0:
            raise ValueError("Time intervals cannot be negative.")
        if self.part_min_common < 2 or self.part_confirmation_hits < 3 or self.part_gallery_size < 2:
            raise ValueError("Partial matching needs two groups, three confirmations and two gallery views.")
        if (self.face_confirmation_hits < 2 or self.face_gallery_size < 2
                or not 0 <= self.face_new_threshold < self.face_match_threshold <= 1):
            raise ValueError("Face matching needs valid similarity thresholds and two confirmations.")


@dataclass
class Observation:
    raw_id: int
    box: tuple[float, float, float, float]
    confidence: float
    embedding: np.ndarray | None = None
    quality: float = 1.0
    part_embeddings: np.ndarray | None = None
    part_visibility: np.ndarray | None = None
    allow_enrollment: bool = True
    face_embedding: np.ndarray | None = None
    face_quality: float = 0.0
    face_sampled: bool = False
    face_status: str = ""


@dataclass(frozen=True)
class CandidateDiagnostic:
    person_id: int
    face: float | None
    body: float | None
    parts: float | None
    score: float | None
    notes: tuple[str, ...]
    part_details: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class TrackDiagnostic:
    timestamp: float
    previous_person_id: int | None
    reason: str
    sampled: bool
    face_used: bool
    body_used: bool
    parts_used: bool
    visible_parts: tuple[int, ...]
    confidence: float
    quality: float
    face_quality: float
    candidates: tuple[CandidateDiagnostic, ...]
    face_sampled: bool = False
    box_changed: bool = False
    gap_seconds: float = 0.0
    face_status: str = ""


@dataclass
class Assignment:
    raw_id: int
    person_id: int | None
    status: str
    similarity: float | None = None
    diagnostic: TrackDiagnostic | None = None


@dataclass
class _Track:
    last_seen: float
    person_id: int | None = None
    needs_verification: bool = False
    view_learning_blocked: bool = False
    first_good: float | None = None
    good_hits: int = 0
    last_embedding_at: float = -math.inf
    candidate_id: int | None = None
    candidate_hits: int = 0
    candidate_last_good: float = -math.inf
    last_box: tuple[float, float, float, float] | None = None
    novel_features: list[np.ndarray] = field(default_factory=list)
    pending_views: list[np.ndarray] = field(default_factory=list)
    pending_since: float | None = None
    novel_parts: list = field(default_factory=list)
    pending_parts: list = field(default_factory=list)
    pending_parts_since: float | None = None
    excluded_ids: set[int] = field(default_factory=set)
    novel_faces: list[np.ndarray] = field(default_factory=list)
    pending_faces: list[np.ndarray] = field(default_factory=list)
    pending_face_since: float | None = None
    pending_face_last: float = -math.inf


@dataclass
class _PartSample:
    embeddings: np.ndarray
    visibility: np.ndarray


@dataclass
class _Identity:
    last_seen: float
    gallery: list[np.ndarray] = field(default_factory=list)
    last_added: float = -math.inf
    part_gallery: list[_PartSample] = field(default_factory=list)
    last_part_added: float = -math.inf
    cannot_link: set[int] = field(default_factory=set)


class IdentityTracker:
    """Keep global IDs across track fragments without expiring absent people.

    Call ``update`` with every visible observation and a monotonic timestamp.
    Send a new embedding about every 0.2 seconds. Pass ``embedding=None``
    between embedding updates or when the crop is unsuitable. Quality gates
    gallery updates, while short track continuity can bridge poor frames.
    """

    def __init__(self, config: TrackerConfig | None = None, *, diagnostics=False):
        self.config = config or TrackerConfig()
        self.diagnostics = diagnostics
        self.identities: dict[int, _Identity] = {}
        self.tracks: dict[int, _Track] = {}
        self.next_id = 1
        self.embedding_dimension: int | None = None
        self.part_shape: tuple[int, int] | None = None
        self.faces = FaceGallery(self.config.face_min_quality, self.config.face_gallery_size,
                                 self.config.gallery_interval, self.config.gallery_similarity)
        self.last_update = -math.inf

    def _embedding(self, observation):
        if (observation.embedding is None or observation.quality < self.config.min_quality
                or observation.confidence < self.config.min_confidence):
            return None
        feature = np.asarray(observation.embedding, dtype=np.float32)
        if feature.ndim != 1 or not np.isfinite(feature).all():
            return None
        norm = float(np.linalg.norm(feature))
        if norm < 1e-8:
            return None
        if self.embedding_dimension is None:
            self.embedding_dimension = feature.size
        if feature.size != self.embedding_dimension:
            raise ValueError("Person embedding dimension changed during the session.")
        return feature / norm

    def _parts(self, observation):
        if (observation.part_embeddings is None or observation.part_visibility is None
                or observation.confidence < self.config.min_confidence):
            return None
        embeddings = np.asarray(observation.part_embeddings, dtype=np.float32)
        visibility = np.asarray(observation.part_visibility, dtype=np.float32)
        if (embeddings.ndim != 2 or min(embeddings.shape, default=0) <= 0
                or visibility.shape != (embeddings.shape[0],)):
            return None
        norms = np.linalg.norm(embeddings, axis=1)
        visible = (np.isfinite(visibility) & (visibility >= self.config.part_min_visibility)
                   & np.isfinite(embeddings).all(axis=1) & (norms > 1e-8))
        if not visible.any():
            return None
        if self.part_shape is None:
            self.part_shape = embeddings.shape
        if embeddings.shape != self.part_shape:
            raise ValueError("Part embedding shape changed during the session.")
        normalized = np.zeros_like(embeddings)
        normalized[visible] = embeddings[visible] / norms[visible, None]
        return _PartSample(normalized, np.where(visible, np.clip(visibility, 0, 1), 0))

    def _group_count(self, visible):
        groups = (self.config.part_groups if len(visible) == len(self.config.part_groups)
                  else tuple(range(len(visible))))
        return len({group for group, valid in zip(groups, visible) if valid})

    def _known_part_support(self, query, identity_ids):
        if query is None:
            return False
        supported = np.zeros(len(query.visibility), dtype=bool)
        for person_id in identity_ids:
            for sample in self.identities[person_id].part_gallery:
                shared = ((query.visibility >= self.config.part_min_visibility)
                          & (sample.visibility >= self.config.part_min_visibility))
                cosines = np.sum(query.embeddings * sample.embeddings, axis=1)
                supported |= shared & (cosines >= self.config.part_match_threshold)
        return self._group_count(supported) >= self.config.part_min_common

    @staticmethod
    def _spatially_distinct(first, second):
        widths = (first[2] - first[0], second[2] - second[0])
        heights = (first[3] - first[1], second[3] - second[1])
        if min(*widths, *heights) <= 0:
            return False
        overlap = (max(0, min(first[2], second[2]) - max(first[0], second[0]))
                   * max(0, min(first[3], second[3]) - max(first[1], second[1])))
        smaller_area = min(widths[0] * heights[0], widths[1] * heights[1])
        dx = abs(first[0] + first[2] - second[0] - second[2]) / 2
        dy = abs(first[1] + first[3] - second[1] - second[3]) / 2
        return (overlap / smaller_area <= 0.15
                and (dx >= 0.75 * min(widths) or dy >= 0.75 * min(heights)))

    def _link_distinct(self, first_id, second_id):
        if first_id != second_id:
            self.identities[first_id].cannot_link.add(second_id)
            self.identities[second_id].cannot_link.add(first_id)

    @staticmethod
    def _duplicate_geometry(first, second):
        width = min(first[2] - first[0], second[2] - second[0])
        height = min(first[3] - first[1], second[3] - second[1])
        if min(width, height) <= 0:
            return True
        overlap = (max(0, min(first[2], second[2]) - max(first[0], second[0]))
                   * max(0, min(first[3], second[3]) - max(first[1], second[1])))
        areas = ((first[2] - first[0]) * (first[3] - first[1]),
                 (second[2] - second[0]) * (second[3] - second[1]))
        return (overlap / max(1, sum(areas) - overlap) >= 0.85
                or (overlap / min(areas) >= 0.92
                    and abs(first[1] - second[1]) <= 0.1 * height
                    and abs(first[0] + first[2] - second[0] - second[2]) <= 0.3 * width))

    def _part_similarity(self, query, gallery, min_groups=None):
        visible = ((query.visibility >= self.config.part_min_visibility)
                   & (gallery.visibility >= self.config.part_min_visibility))
        count = self._group_count(visible)
        if count < (self.config.part_min_common if min_groups is None else min_groups):
            return -1.0
        weights = np.minimum(query.visibility[visible], gallery.visibility[visible])
        cosines = np.sum(query.embeddings[visible] * gallery.embeddings[visible], axis=1)
        # Paired limbs count as one body group, including in the average.
        groups = (self.config.part_groups if len(visible) == len(self.config.part_groups)
                  else tuple(range(len(visible))))
        group_scores = []
        for group in {groups[index] for index in np.flatnonzero(visible)}:
            selected = np.asarray([groups[index] == group for index in np.flatnonzero(visible)])
            group_scores.append(float(np.average(cosines[selected], weights=weights[selected])))
        return float(np.mean(group_scores))

    def _part_scores(self, parts, identity_ids):
        scores = np.full((len(parts), len(identity_ids)), -1.0, dtype=np.float32)
        raw_scores = np.full_like(scores, np.nan) if self.diagnostics else None
        best_samples = {}
        bridges = scores.copy()
        compatible = scores.copy()
        conflicts = np.zeros_like(scores, dtype=bool)
        known_part = np.zeros(len(parts), dtype=bool)
        for row, query in enumerate(parts):
            if query is None:
                continue
            local = np.full((len(identity_ids), query.embeddings.shape[0]), -1.0)
            for column, person_id in enumerate(identity_ids):
                gallery = self.identities[person_id].part_gallery
                for sample in gallery:
                    shared = ((query.visibility >= self.config.part_min_visibility)
                              & (sample.visibility >= self.config.part_min_visibility))
                    cosines = np.sum(query.embeddings * sample.embeddings, axis=1)
                    local[column, shared] = np.maximum(local[column, shared], cosines[shared])
                raw, best_sample = max(((self._part_similarity(query, old), old) for old in gallery),
                                       key=lambda result: result[0], default=(-1.0, None))
                if raw > -1:
                    scores[row, column] = self.config.match_threshold + raw - self.config.part_match_threshold
                    if raw_scores is not None:
                        raw_scores[row, column] = raw
                        best_samples[row, column] = best_sample
                one, bridge_sample = max(((self._part_similarity(query, old, min_groups=1), old)
                                          for old in gallery), key=lambda result: result[0],
                                         default=(-1.0, None))
                if one >= self.config.part_match_threshold:
                    bridges[row, column] = self.config.match_threshold + one - self.config.part_match_threshold
                    if raw_scores is not None and raw == -1.0:
                        raw_scores[row, column] = one
                        best_samples[row, column] = bridge_sample
            if identity_ids:
                known_part[row] = (self._group_count(
                    (local >= self.config.part_match_threshold).any(axis=0))
                                   >= self.config.part_min_common)
            comparison_scores = scores[row].copy()
            comparison_bridges = bridges[row].copy()
            compatible[row] = np.maximum(comparison_scores, comparison_bridges)
            for column in range(len(identity_ids)):
                others = np.delete(local, column, axis=0)
                if others.size:
                    rival = others.max(axis=0)
                    conflicts[row, column] = bool(np.any(
                        (rival >= self.config.part_match_threshold)
                        & (rival - local[column] >= self.config.part_ambiguity_margin)))
                    if conflicts[row, column]:
                        scores[row, column] = -1.0
                        bridges[row, column] = -1.0
            # Calibrate and reject ambiguous partial evidence before it can
            # compete with the separately calibrated global encoder.
            for matrix, original in ((scores, comparison_scores), (bridges, comparison_bridges)):
                for column, score in enumerate(original):
                    rivals = np.delete(original, column)
                    if (score >= self.config.match_threshold and rivals.size
                            and score - float(rivals.max()) < self.config.part_ambiguity_margin):
                        matrix[row, column] = -1.0
        return scores, bridges, conflicts, known_part, compatible, raw_scores, best_samples

    def _remember(self, identity, feature, now, similarity, *, trusted=False, force=False):
        if feature is None or (not force and now - identity.last_added < self.config.gallery_interval):
            return
        if identity.gallery:
            if not trusted and similarity is not None and similarity < self.config.match_threshold:
                return
            if max(float(feature @ old) for old in identity.gallery) >= self.config.gallery_similarity:
                return
        if len(identity.gallery) >= self.config.gallery_size:
            # Retain unusual old angles instead of FIFO-erasing every view
            # except the first one. Prefer dropping the newer redundant view.
            candidates = identity.gallery + [feature]
            matrix = np.stack(candidates) @ np.stack(candidates).T
            np.fill_diagonal(matrix, -np.inf)
            redundancy = matrix.max(axis=1)
            drop = max(range(1, len(candidates)), key=lambda index: (redundancy[index], index))
            if drop == len(candidates) - 1:
                return
            identity.gallery.pop(drop)
        identity.gallery.append(feature.copy())
        identity.last_added = now

    @staticmethod
    def _clear_pending(track, *, parts=True):
        track.pending_views.clear()
        track.pending_since = None
        if parts:
            track.pending_parts.clear()
            track.pending_parts_since = None

    @staticmethod
    def _clear_novel(track):
        track.first_good = None
        track.good_hits = 0
        track.novel_features.clear()
        track.novel_parts.clear()
        track.novel_faces.clear()

    @staticmethod
    def _clear_face_pending(track):
        track.pending_faces.clear()
        track.pending_face_since = None

    def _learn_face(self, track, person_id, feature, now, confirmed):
        if feature is None:
            return
        if self.faces.galleries.get(person_id):
            if confirmed:
                self.faces.remember(person_id, feature, now)
            self._clear_face_pending(track)
            return
        if (now - track.pending_face_last > self.config.candidate_memory_seconds
                or (track.pending_faces and min(float(old @ feature) for old in track.pending_faces)
                    < self.config.face_match_threshold)):
            self._clear_face_pending(track)
        if not track.pending_faces:
            track.pending_face_since = now
        track.pending_faces.append(feature.copy())
        track.pending_face_last = now
        if (len(track.pending_faces) >= self.config.view_confirmation_hits
                and now - track.pending_face_since + 1e-9 >= self.config.view_confirmation_seconds):
            for sample in track.pending_faces:
                self.faces.remember(person_id, sample, now, force=True)
            self._clear_face_pending(track)

    def _remember_parts(self, identity, sample, now, *, force=False):
        if sample is None or self._group_count(sample.visibility >= self.config.part_min_visibility) < self.config.part_min_common:
            return
        if not force and now - identity.last_part_added < self.config.gallery_interval:
            return
        if any(self._part_similarity(sample, old) >= self.config.gallery_similarity
               and np.all((sample.visibility >= self.config.part_min_visibility)
                          <= (old.visibility >= self.config.part_min_visibility))
               for old in identity.part_gallery):
            return
        if len(identity.part_gallery) >= self.config.part_gallery_size:
            candidates = identity.part_gallery + [sample]
            redundancy = [max(self._part_similarity(item, other)
                              for j, other in enumerate(candidates) if j != index)
                          for index, item in enumerate(candidates)]
            drop = max(range(1, len(candidates)), key=lambda index: (redundancy[index], index))
            if drop == len(candidates) - 1:
                return
            identity.part_gallery.pop(drop)
        identity.part_gallery.append(_PartSample(sample.embeddings.copy(), sample.visibility.copy()))
        identity.last_part_added = now

    def _learn_parts(self, track, identity, sample, now):
        if (track.view_learning_blocked or sample is None
                or self._group_count(sample.visibility >= self.config.part_min_visibility) < self.config.part_min_common):
            track.pending_parts.clear()
            track.pending_parts_since = None
            return
        if not track.pending_parts:
            track.pending_parts_since = now
        track.pending_parts.append(sample)
        track.pending_parts = track.pending_parts[-max(self.config.part_gallery_size,
                                                     self.config.view_confirmation_hits):]
        if (len(track.pending_parts) < self.config.view_confirmation_hits
                or now - track.pending_parts_since + 1e-9 < self.config.view_confirmation_seconds):
            return
        visibility = np.min(np.stack([old.visibility for old in track.pending_parts]), axis=0)
        visible = visibility >= self.config.part_min_visibility
        if self._group_count(visible) >= self.config.part_min_common:
            pooled = np.mean(np.stack([old.embeddings for old in track.pending_parts]), axis=0)
            norms = np.linalg.norm(pooled, axis=1)
            valid = visible & (norms > 1e-8)
            normalized = np.zeros_like(pooled)
            normalized[valid] = pooled[valid] / norms[valid, None]
            coherent = (self._group_count(valid) >= self.config.part_min_common
                        and all(np.all(np.sum(old.embeddings[valid] * normalized[valid], axis=1)
                                       >= self.config.view_consistency_threshold)
                                for old in track.pending_parts))
            if coherent:
                self._remember_parts(identity, _PartSample(normalized, np.where(valid, visibility, 0)), now)
        track.pending_parts.clear()
        track.pending_parts_since = None

    @staticmethod
    def _plausible_motion(previous, current):
        """Gate view learning locally; positions never gate reidentification."""
        if previous is None:
            return True
        widths = (previous[2] - previous[0], current[2] - current[0])
        heights = (previous[3] - previous[1], current[3] - current[1])
        if min(*widths, *heights) <= 0:
            return False
        if max(widths) / min(widths) > 1.8 or max(heights) / min(heights) > 1.8:
            return False
        dx = abs((previous[0] + previous[2]) - (current[0] + current[2])) / 2
        dy = abs((previous[1] + previous[3]) - (current[1] + current[3])) / 2
        return dx <= 0.5 * max(widths) and dy <= 0.5 * max(heights)

    def _learn_continuous_view(self, track, identity, observation, feature, now,
                               similarity, best_other, *, face_verified=False):
        """Stage weak new angles only while a clean raw track stays bound.

        A weak observation cannot identify an absent person. Several consistent
        observations on an uninterrupted track can instead teach a new view for
        a later return, provided no other identity competes for that view.
        """
        if (track.view_learning_blocked
                or observation.quality < self.config.view_min_quality
                or (not face_verified and similarity - best_other < self.config.ambiguity_margin)):
            self._clear_pending(track, parts=False)
            return
        if not track.pending_views:
            track.pending_since = now
        track.pending_views.append(feature.copy())
        track.pending_views = track.pending_views[-max(self.config.gallery_size,
                                                      self.config.view_confirmation_hits):]
        if (len(track.pending_views) < self.config.view_confirmation_hits
                or now - track.pending_since + 1e-9 < self.config.view_confirmation_seconds):
            return
        representative = np.mean(track.pending_views, axis=0)
        norm = float(np.linalg.norm(representative))
        if norm < 1e-8:
            self._clear_pending(track, parts=False)
            return
        representative /= norm
        # Compare every sample with the pooled view. Gait changes make even
        # adjacent genuine crops noisier than a coherent multi-frame view.
        if (min(float(representative @ old) for old in track.pending_views)
                < self.config.view_consistency_threshold):
            self._clear_pending(track, parts=False)
            return
        own_score = max((float(representative @ old) for old in identity.gallery), default=-1.0)
        other_score = max((float(representative @ old)
                           for other in self.identities.values() if other is not identity
                           for old in other.gallery), default=-1.0)
        if (face_verified or (own_score >= self.config.continuity_min_similarity
                              and own_score - other_score >= self.config.ambiguity_margin)):
            self._remember(identity, representative, now, own_score, trusted=True)
        self._clear_pending(track, parts=False)

    def _scores(self, features, identity_ids):
        scores = np.full((len(features), len(identity_ids)), -1.0, dtype=np.float32)
        for row, feature in enumerate(features):
            if feature is None:
                continue
            for column, person_id in enumerate(identity_ids):
                gallery = self.identities[person_id].gallery
                if gallery:
                    scores[row, column] = float(np.max(np.stack(gallery) @ feature))
        return scores

    def update(self, observations: list[Observation], now: float) -> list[Assignment]:
        if not math.isfinite(now) or now < self.last_update:
            raise ValueError("Use finite, monotonic timestamps for identity association.")
        if len({observation.raw_id for observation in observations}) != len(observations):
            raise ValueError("A temporary tracker ID appeared more than once in one frame.")
        self.last_update = now
        stale_after = max(5.0, 3 * self.config.continuity_seconds, 3 * self.config.new_identity_seconds)
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.last_seen <= stale_after}
        visible = {observation.raw_id for observation in observations}
        for raw_id, track in self.tracks.items():
            if raw_id not in visible:
                track.view_learning_blocked = True
                self._clear_pending(track)
                self._clear_face_pending(track)
                self._clear_novel(track)
                track.candidate_id = None
                track.candidate_hits = 0
        if not observations:
            return []

        features = [self._embedding(observation) for observation in observations]
        parts = [self._parts(observation) for observation in observations]
        faces = [self.faces.feature(observation.face_embedding, observation.face_quality)
                 if observation.confidence >= self.config.min_confidence else None
                 for observation in observations]
        face_received = [face is not None for face in faces]
        has_evidence = [feature is not None or sample is not None or face is not None
                        for feature, sample, face in zip(features, parts, faces)]
        enrollment_evidence = [observation.allow_enrollment and (
            feature is not None or face is not None or (sample is not None and self._group_count(
                sample.visibility >= self.config.part_min_visibility) >= self.config.part_min_common))
                               for observation, feature, sample, face in zip(observations, features, parts, faces)]
        previous_seen = []
        previous_bindings = []
        box_changes = []
        fresh_features = []
        for row, (observation, feature) in enumerate(zip(observations, features)):
            track = self.tracks.setdefault(observation.raw_id, _Track(now))
            previous_seen.append(track.last_seen)
            previous_bindings.append(track.person_id)
            if not observation.allow_enrollment:
                self._clear_novel(track)
            gap = now - track.last_seen > self.config.continuity_seconds
            moved = not self._plausible_motion(track.last_box, observation.box)
            box_changes.append(moved)
            if gap or moved:
                self._clear_pending(track)
                self._clear_face_pending(track)
                self._clear_novel(track)
                track.candidate_id = None
                track.candidate_hits = 0
                track.view_learning_blocked = True
                if gap and track.person_id is not None:
                    track.needs_verification = True
                if gap:
                    track.excluded_ids.clear()
            if now - track.candidate_last_good > self.config.candidate_memory_seconds:
                track.candidate_id = None
                track.candidate_hits = 0
            if (observation.quality < self.config.view_min_quality
                    or observation.confidence < self.config.min_confidence):
                self._clear_pending(track, parts=False)
            if ((observation.part_embeddings is not None and parts[row] is None)
                    or observation.confidence < self.config.min_confidence):
                track.pending_parts.clear()
                track.pending_parts_since = None
            track.last_seen = now
            track.last_box = observation.box
            fresh = has_evidence[row] and now > track.last_embedding_at
            fresh_features.append(fresh)
            if fresh:
                track.last_embedding_at = now

        identity_ids = list(self.identities)
        global_scores = self._scores(features, identity_ids)
        (part_scores, part_bridges, part_conflicts, known_part, part_compatible,
         raw_part_scores, best_part_samples) = self._part_scores(parts, identity_ids)
        scores = np.maximum(global_scores, part_scores)
        continuity_scores = np.maximum(global_scores, part_compatible)
        initial_face_scores = self.faces.scores(faces, identity_ids)
        for row, observation in enumerate(observations):
            track = self.tracks[observation.raw_id]
            if (faces[row] is None or track.person_id not in self.identities
                    or self.faces.galleries.get(track.person_id) or track.needs_verification
                    or now - previous_seen[row] > self.config.continuity_seconds):
                continue
            own_column = identity_ids.index(track.person_id)
            if continuity_scores[row, own_column] < self.config.match_threshold:
                continue
            # A first face observation must not erase an established body-only
            # identity by claiming a person already proven distinct from it.
            for column, person_id in enumerate(identity_ids):
                if (person_id in self.identities[track.person_id].cannot_link
                        and initial_face_scores[row, column] >= self.config.face_match_threshold):
                    faces[row] = None
                    self._clear_face_pending(track)
                    break
        scores, continuity_scores, face_scores, face_confirmed = self.faces.fuse(
            faces, identity_ids, scores, continuity_scores, self.config.face_match_threshold,
            self.config.face_ambiguity_margin, self.config.face_new_threshold, self.config.match_threshold)
        # Resolve trusted owners first. A merely stronger lookalike must not
        # outbid a compatible visible owner through a small assignment bonus.
        continuity = np.zeros_like(scores, dtype=bool)
        owners = {}
        for row, observation in enumerate(observations):
            track = self.tracks[observation.raw_id]
            for column, person_id in enumerate(identity_ids):
                score = float(scores[row, column])
                continuous = (track.person_id == person_id
                              and not track.needs_verification
                              and now - previous_seen[row] <= self.config.continuity_seconds)
                if continuous:
                    score = float(continuity_scores[row, column])
                if continuous and has_evidence[row]:
                    others = np.delete(scores[row], column)
                    best_other = float(others.max()) if others.size else -1.0
                    continuous = (score >= self.config.continuity_min_similarity
                                  and not (score < self.config.match_threshold
                                           and best_other >= self.config.match_threshold
                                           and best_other - score >= self.config.ambiguity_margin))
                    if not continuous:
                        # Between embedding samples, keep this binding quarantined
                        # until fresh appearance evidence confirms an identity.
                        track.needs_verification = True
                        self._clear_pending(track)
                        # This raw number may now describe another person.
                        track.excluded_ids.clear()
                continuity[row, column] = continuous
                if continuous:
                    owners[column] = row

        clean = [observation.confidence >= self.config.min_confidence and (
            (feature is not None and observation.quality >= self.config.view_min_quality)
            or face is not None
            or (sample is not None and self._group_count(
                sample.visibility >= self.config.part_min_visibility) >= self.config.part_min_common))
                 for observation, feature, sample, face in zip(observations, features, parts, faces)]
        for column, owner_row in owners.items():
            owner_track = self.tracks[observations[owner_row].raw_id]
            owner_recent = now - owner_track.last_embedding_at <= self.config.candidate_memory_seconds
            if not owner_recent and observations[owner_row].confidence < self.config.min_confidence:
                continue
            person_id = identity_ids[column]
            for row, observation in enumerate(observations):
                if (row == owner_row or not clean[row]
                        or not self._spatially_distinct(observation.box, observations[owner_row].box)):
                    continue
                track = self.tracks[observation.raw_id]
                track.excluded_ids.add(person_id)
                if track.candidate_id == person_id:
                    track.candidate_id = None
                    track.candidate_hits = 0

        association_scores = scores.copy()
        for row, observation in enumerate(observations):
            track = self.tracks[observation.raw_id]
            for column, person_id in enumerate(identity_ids):
                if person_id in track.excluded_ids:
                    association_scores[row, column] = -1.0
        enrollment_geometry = [not any(
            owner_row != row and self._duplicate_geometry(observation.box, observations[owner_row].box)
            for owner_row in owners.values()) for row, observation in enumerate(observations)]
        costs = np.full((len(observations), len(identity_ids) + len(observations)), 0.0)
        costs[:, :len(identity_ids)] = 1e6
        reserved_rows = set(owners.values())
        for row in range(len(observations)):
            for column in range(len(identity_ids)):
                if column in owners:
                    if owners[column] == row:
                        costs[row, column] = -1.0
                elif row not in reserved_rows and association_scores[row, column] >= self.config.match_threshold:
                    costs[row, column] = -float(association_scores[row, column])

        rows, columns = linear_sum_assignment(costs)
        proposed = dict(zip(rows.tolist(), columns.tolist()))
        assignments = []
        accepted_ids = set()
        if self.diagnostics:
            face_gallery_ids = {key for key in identity_ids if self.faces.galleries.get(key)}
            body_gallery_ids = {key for key in identity_ids if self.identities[key].gallery}
        for row, observation in enumerate(observations):
            track = self.tracks[observation.raw_id]
            column = proposed[row]
            person_id = identity_ids[column] if column < len(identity_ids) else None
            similarity = (float(continuity_scores[row, column] if continuity[row, column]
                                else scores[row, column])
                          if person_id is not None and has_evidence[row] else None)
            status = "checking"
            matching_candidate = False
            reason = "비교할 새 특징을 기다리는 중"

            if person_id is not None and continuity[row, column]:
                status = "tracked"
                reason = ("얼굴 일치 · 기존 ID 유지" if face_confirmed[row, column]
                          else "기존 궤적과 외형 단서로 ID 유지" if has_evidence[row]
                          else "기존 궤적으로 ID 유지 · 새 비교 대기")
            elif person_id is not None:
                others = np.delete(association_scores[row], column)
                best_other = float(others.max()) if others.size else -1.0
                if similarity is not None and similarity - best_other >= self.config.ambiguity_margin:
                    matching_candidate = True
                    if fresh_features[row]:
                        if track.candidate_id == person_id:
                            track.candidate_hits += 1
                        else:
                            track.candidate_id = person_id
                            track.candidate_hits = 1
                        track.candidate_last_good = now
                    part_only = global_scores[row, column] < self.config.match_threshold
                    needed_hits = (self.config.face_confirmation_hits if face_confirmed[row, column]
                                   else max(self.config.confirmation_hits, self.config.part_confirmation_hits)
                                   if part_only else self.config.confirmation_hits)
                    if face_confirmed[row, column]:
                        source = "얼굴"
                    elif part_scores[row, column] > global_scores[row, column]:
                        source = "공통 신체 부위"
                    else:
                        source = "박스 전체 외형"
                    reason = f"{source} 후보 P{person_id:03d} 확인 중 · {track.candidate_hits}/{needed_hits}회"
                    if track.candidate_hits >= needed_hits:
                        status = "reidentified"
                        reason = f"{source} 일치 · P{person_id:03d} 재연결"
                else:
                    track.candidate_id = None
                    track.candidate_hits = 0
                    reason = "후보 간 점수 차가 작아 보류"

            if status == "checking":
                self._clear_pending(track)
                self._clear_face_pending(track)
                person_id = None
                if fresh_features[row] and not matching_candidate:
                    candidate_column = (identity_ids.index(track.candidate_id)
                                        if track.candidate_id in identity_ids else None)
                    candidate_score = (float(association_scores[row, candidate_column])
                                       if candidate_column is not None else -1.0)
                    # A brief inconclusive angle may separate strong votes.
                    # It never counts as a vote or overrides a strong rival.
                    keep_candidate = (
                        candidate_column is not None
                        and now - track.candidate_last_good <= self.config.candidate_memory_seconds
                        and candidate_score >= self.config.continuity_min_similarity
                        and float(association_scores[row].max()) < self.config.match_threshold
                    )
                    if not keep_candidate:
                        track.candidate_id = None
                        track.candidate_hits = 0
                best_score = float(association_scores[row].max()) if identity_ids else -1.0
                novelty_margin = (self.config.part_ambiguity_margin
                                  if features[row] is None and parts[row] is not None
                                  else self.config.ambiguity_margin)
                clearly_new = best_score < self.config.match_threshold - novelty_margin
                if fresh_features[row]:
                    available_ids = [person_id for person_id in identity_ids
                                     if person_id not in track.excluded_ids]
                    available_known_part = (self._known_part_support(parts[row], available_ids)
                                            if track.excluded_ids else known_part[row])
                    valid_novel = (enrollment_evidence[row]
                                   and enrollment_geometry[row]
                                   and not (track.person_id is not None and face_confirmed[row].any())
                                   and (features[row] is not None or faces[row] is not None
                                        or not available_known_part))
                    if clearly_new and track.candidate_id is None and valid_novel:
                        feature = features[row]
                        sample = parts[row]
                        face = faces[row]
                        if (face is not None and track.novel_faces
                                and min(float(face @ old) for old in track.novel_faces)
                                < self.config.face_match_threshold):
                            self._clear_novel(track)
                        if (feature is not None and track.novel_features
                                and max(float(feature @ old) for old in track.novel_features)
                                < self.config.continuity_min_similarity):
                            self._clear_novel(track)
                        if (feature is None and sample is not None and track.novel_parts
                                and max(self._part_similarity(sample, old) for old in track.novel_parts)
                                < self.config.continuity_min_similarity):
                            self._clear_novel(track)
                        if track.first_good is None:
                            track.first_good = now
                        track.good_hits += 1
                        if feature is not None:
                            track.novel_features.append(feature.copy())
                            track.novel_features = track.novel_features[-self.config.gallery_size:]
                        if sample is not None:
                            track.novel_parts.append(sample)
                            track.novel_parts = track.novel_parts[-self.config.part_gallery_size:]
                        if face is not None:
                            track.novel_faces.append(face.copy())
                            track.novel_faces = track.novel_faces[-self.config.face_gallery_size:]
                        reason = (f"새 ID 확인 중 · {track.good_hits}/{self.config.new_identity_hits}회"
                                  f" · {now - track.first_good:.1f}/{self.config.new_identity_seconds:.1f}초")
                    else:
                        # Known or ambiguous evidence invalidates a run of
                        # novelty; a single later weak crop cannot mint an ID.
                        self._clear_novel(track)
                        if not matching_candidate:
                            if not enrollment_geometry[row]:
                                reason = "기존 박스와 겹쳐 새 ID 등록 보류"
                            elif not enrollment_evidence[row]:
                                reason = "새 ID 등록에 필요한 얼굴 또는 신체 단서 부족"
                            elif not clearly_new:
                                reason = "비슷한 기존 후보가 있어 새 ID 등록 보류"
                            else:
                                reason = "기존 ID와 충돌하는 단서가 있어 판정 보류"
                ready = (track.first_good is not None
                         and now - track.first_good >= self.config.new_identity_seconds
                         and track.good_hits >= self.config.new_identity_hits)
                if (fresh_features[row] and enrollment_evidence[row] and enrollment_geometry[row]
                        and clearly_new and ready):
                    person_id = self.next_id
                    self.next_id += 1
                    self.identities[person_id] = _Identity(now)
                    for other_id in track.excluded_ids:
                        self._link_distinct(person_id, other_id)
                    status = "new"
                    reason = (f"{'기존 후보와 연결되지 않아 새 ID' if identity_ids else '첫 ID'} 등록"
                              f" · {track.good_hits}회 관측, {now - track.first_good:.1f}초")

            diagnostic = None
            if self.diagnostics:
                candidates = []
                sample = parts[row]
                face_matched = bool(face_confirmed[row].any())
                best_face = max((float(initial_face_scores[row, index])
                                 for index, key in enumerate(identity_ids) if key in face_gallery_ids),
                                default=-1.0)
                for index, key in enumerate(identity_ids):
                    face = (float(initial_face_scores[row, index])
                            if face_received[row] and key in face_gallery_ids else None)
                    body = (float(global_scores[row, index])
                            if features[row] is not None and key in body_gallery_ids else None)
                    part = raw_part_scores[row, index]
                    part = float(part) if np.isfinite(part) else None
                    if face is None and body is None and part is None:
                        continue
                    notes = []
                    best_sample = best_part_samples.get((row, index))
                    part_details = ()
                    bridge_only = False
                    if best_sample is not None:
                        shared = ((sample.visibility >= self.config.part_min_visibility)
                                  & (best_sample.visibility >= self.config.part_min_visibility))
                        bridge_only = self._group_count(shared) < self.config.part_min_common
                        part_details = tuple((int(part_index), float(sample.embeddings[part_index]
                                                                   @ best_sample.embeddings[part_index]))
                                             for part_index in np.flatnonzero(shared))
                    if face is not None:
                        if faces[row] is None:
                            notes.append("별도 인물 기록으로 얼굴 보류")
                        elif face_confirmed[row, index]:
                            notes.append("얼굴 일치")
                        elif face_matched:
                            notes.append("다른 후보와 얼굴 일치")
                        elif best_face < self.config.face_new_threshold:
                            notes.append("얼굴 불일치로 제외")
                        else:
                            notes.append("얼굴 비교 보류")
                    elif face_received[row] and key not in face_gallery_ids:
                        notes.append("이 ID의 등록 얼굴 없음")
                    if part_conflicts[row, index]:
                        notes.append("부위 단서 충돌")
                    elif (not bridge_only and part is not None and part >= self.config.part_match_threshold
                          and part_scores[row, index] == -1.0):
                        notes.append("부위 후보 간 점수 차 부족")
                    if bridge_only:
                        notes.append("공통 부위 한 그룹 · 연속 추적용")
                    if key in track.excluded_ids:
                        notes.append("같은 화면의 별도 인물")
                    elif index in owners and owners[index] != row:
                        notes.append("다른 박스에 배정됨")
                    score = float(continuity_scores[row, index] if continuity[row, index]
                                  else association_scores[row, index])
                    candidates.append(CandidateDiagnostic(key, face, body, part,
                                                          score if score > -1 else None, tuple(notes), part_details))
                previous_id = previous_bindings[row]
                candidates.sort(key=lambda candidate: (
                    candidate.person_id not in (previous_id, person_id),
                    -max(value for value in (candidate.face, candidate.body, candidate.parts, candidate.score)
                         if value is not None)))
                visible_parts = (tuple(int(index) for index in np.flatnonzero(
                    sample.visibility >= self.config.part_min_visibility)) if sample is not None else ())
                if not has_evidence[row] and observation.face_sampled and status == "checking":
                    reason = "얼굴 비교를 시도했지만 쓸 수 있는 특징이 없음"
                face_status = observation.face_status
                if face_received[row]:
                    if faces[row] is None:
                        face_status = "얼굴 단서 보류"
                    else:
                        face_status = "비교 완료" if face_gallery_ids else "등록 얼굴 없음"
                elif observation.face_embedding is not None:
                    face_status = "얼굴 비교 품질 기준 미달"
                diagnostic = TrackDiagnostic(
                    now, previous_id, reason, has_evidence[row], faces[row] is not None,
                    features[row] is not None, sample is not None, visible_parts,
                    observation.confidence, observation.quality, observation.face_quality,
                    tuple(candidates[:6]), observation.face_sampled, box_changes[row],
                    now - previous_seen[row], face_status)

            if person_id is not None:
                if person_id in accepted_ids:
                    raise RuntimeError("Two observations were assigned the same person ID.")
                accepted_ids.add(person_id)
                # Release old raw bindings after reacquisition. An old tracker
                # fragment may return later and must compete on its own merit.
                if status == "reidentified":
                    for old_raw, old_track in self.tracks.items():
                        if old_raw != observation.raw_id and old_track.person_id == person_id:
                            old_track.person_id = None
                            self._clear_pending(old_track)
                track.person_id = person_id
                track.needs_verification = False
                identity = self.identities[person_id]
                identity.last_seen = now
                if status == "new":
                    track.view_learning_blocked = False
                    for feature in track.novel_features:
                        self._remember(identity, feature, now, None, trusted=True, force=True)
                    for sample in track.novel_parts:
                        self._remember_parts(identity, sample, now, force=True)
                    for face in track.novel_faces:
                        self.faces.remember(person_id, face, now, force=True)
                else:
                    others = np.delete(scores[row], column)
                    best_other = float(others.max()) if others.size else -1.0
                    conflicted = part_conflicts[row, column]
                    separated = (similarity is not None
                                 and similarity - best_other >= self.config.ambiguity_margin)
                    if not conflicted and separated:
                        strong_parts = part_scores[row, column] >= self.config.match_threshold
                        strong_global = (features[row] is not None
                                         and observation.quality >= self.config.view_min_quality
                                         and global_scores[row, column] >= self.config.match_threshold)
                        if strong_global or strong_parts or face_confirmed[row, column]:
                            track.view_learning_blocked = False
                        if features[row] is not None:
                            global_similarity = float(global_scores[row, column])
                            global_others = np.delete(global_scores[row], column)
                            global_other = float(global_others.max()) if global_others.size else -1.0
                            if (face_confirmed[row, column]
                                    and global_similarity < self.config.match_threshold):
                                self._learn_continuous_view(track, identity, observation, features[row],
                                                            now, global_similarity, global_other,
                                                            face_verified=True)
                            elif (status == "tracked"
                                    and self.config.continuity_min_similarity <= global_similarity < self.config.match_threshold):
                                self._learn_continuous_view(track, identity, observation, features[row],
                                                            now, global_similarity, global_other)
                            else:
                                self._remember(identity, features[row], now, global_similarity)
                                self._clear_pending(track, parts=False)
                        if fresh_features[row] and parts[row] is not None:
                            self._learn_parts(track, identity, parts[row], now)
                    elif has_evidence[row]:
                        self._clear_pending(track)
                    if fresh_features[row]:
                        if (face_confirmed[row, column]
                                or (not conflicted and not track.view_learning_blocked
                                    and (separated or continuity[row, column]))):
                            self._learn_face(track, person_id, faces[row], now, face_confirmed[row, column])
                        elif faces[row] is not None:
                            self._clear_face_pending(track)
                track.candidate_id = None
                track.candidate_hits = 0
                self._clear_novel(track)
            assignments.append(Assignment(observation.raw_id, person_id, status, similarity, diagnostic))
        for row, assignment in enumerate(assignments):
            if assignment.person_id is None or not clean[row]:
                continue
            for other_row in range(row):
                other = assignments[other_row]
                if (other.person_id is not None and clean[other_row]
                        and self._spatially_distinct(observations[row].box, observations[other_row].box)):
                    self._link_distinct(assignment.person_id, other.person_id)
                    self.tracks[assignment.raw_id].excluded_ids.add(other.person_id)
                    self.tracks[other.raw_id].excluded_ids.add(assignment.person_id)
        return assignments
