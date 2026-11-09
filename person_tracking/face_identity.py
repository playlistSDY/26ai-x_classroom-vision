"""Session-local galleries of normalized, quality-gated face embeddings."""

import math

import numpy as np


class FaceGallery:
    def __init__(self, min_quality=0.7, size=12, interval=0.35, duplicate_similarity=0.98):
        self.min_quality = min_quality
        self.size = size
        self.interval = interval
        self.duplicate_similarity = duplicate_similarity
        self.dimension = None
        self.galleries: dict[int, list[np.ndarray]] = {}
        self.last_added: dict[int, float] = {}

    def feature(self, embedding, quality):
        if embedding is None or not math.isfinite(quality) or quality < self.min_quality:
            return None
        feature = np.asarray(embedding, dtype=np.float32)
        if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
            return None
        norm = float(np.linalg.norm(feature))
        if norm < 1e-8:
            return None
        if self.dimension is None:
            self.dimension = feature.size
        if feature.size != self.dimension:
            raise ValueError("Face embedding dimension changed during the session.")
        return feature / norm

    def scores(self, features, identity_ids):
        scores = np.full((len(features), len(identity_ids)), -1.0, dtype=np.float32)
        for column, person_id in enumerate(identity_ids):
            gallery = self.galleries.get(person_id)
            if not gallery:
                continue
            matrix = np.stack(gallery)
            for row, feature in enumerate(features):
                if feature is not None:
                    scores[row, column] = float(np.max(matrix @ feature))
        return scores

    def fuse(self, features, identity_ids, body_scores, continuity_scores,
             match_threshold, ambiguity_margin, new_threshold, body_threshold):
        """Prioritize a clear face and prevent clothes from bypassing it.

        Unknown/ambiguous face observations may preserve a compatible local
        track, but cannot reacquire a known ID using clothing alone.
        """
        raw = self.scores(features, identity_ids)
        scores = body_scores.copy()
        continuity = continuity_scores.copy()
        confirmed = np.zeros_like(scores, dtype=bool)
        known_columns = np.asarray([bool(self.galleries.get(key)) for key in identity_ids])
        deferred = body_threshold - 0.03
        for row, feature in enumerate(features):
            if feature is None or not known_columns.any():
                continue
            ranked = sorted(np.flatnonzero(known_columns), key=lambda column: raw[row, column], reverse=True)
            best = ranked[0]
            second = float(raw[row, ranked[1]]) if len(ranked) > 1 else -1.0
            best_score = float(raw[row, best])
            if best_score >= match_threshold and best_score - second >= ambiguity_margin:
                scores[row] = -1.0
                continuity[row] = -1.0
                scores[row, best] = continuity[row, best] = 1.1 + 0.1 * best_score
                confirmed[row, best] = True
            elif best_score >= new_threshold:
                # There is a plausible known face but no decisive identity.
                # A high clothes score must not turn this into a positive match.
                scores[row] = np.minimum(scores[row], deferred)
                continuity[row] = np.minimum(continuity[row], deferred)
                scores[row, known_columns] = deferred
                continuity[row, known_columns] = deferred
            else:
                # A clear different face rules out every identity for which
                # a reliable face has already been registered.
                scores[row, known_columns] = -1.0
                continuity[row, known_columns] = -1.0
        return scores, continuity, raw, confirmed

    def remember(self, person_id, feature, now, *, force=False):
        if feature is None or (not force and now - self.last_added.get(person_id, -math.inf) < self.interval):
            return
        gallery = self.galleries.setdefault(person_id, [])
        if gallery and max(float(old @ feature) for old in gallery) >= self.duplicate_similarity:
            return
        if len(gallery) >= self.size:
            candidates = gallery + [feature]
            matrix = np.stack(candidates) @ np.stack(candidates).T
            np.fill_diagonal(matrix, -np.inf)
            redundancy = matrix.max(axis=1)
            drop = max(range(1, len(candidates)), key=lambda index: (redundancy[index], index))
            if drop == len(candidates) - 1:
                return
            gallery.pop(drop)
        gallery.append(feature.copy())
        self.last_added[person_id] = now
