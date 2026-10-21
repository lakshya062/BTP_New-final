"""Conservative, offline multi-view association and local packet serialization.

Local tracker IDs have meaning within one camera frame only. Identity evidence is
never cached or copied to later frames. This module does not alter the live UI's
single-camera worker; callers must supply calibrated, synchronized observations.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .multicamera_geometry import estimate_pair_compatibility, fuse_pose
from .multicamera_types import CalibratedCamera, finite_scalar, immutable_array, validated_identifier


def _identifiers(value, name, *, maximum):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identifiers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of identifiers") from exc
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds its size limit")
    for item in values:
        validated_identifier(item, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain distinct identifiers")
    return values


def _observation_keys(value):
    if isinstance(value, (str, bytes)):
        raise ValueError("observation_keys must be a sequence of four-tuples")
    try:
        keys = tuple(tuple(key) for key in value)
    except TypeError as exc:
        raise ValueError("observation_keys must contain four-tuples") from exc
    if not 1 <= len(keys) <= 8:
        raise ValueError("a fused person needs one to eight observation keys")
    normalized = []
    for key in keys:
        if len(key) != 4:
            raise ValueError("observation keys require camera, epoch, sequence and local track ID")
        camera, epoch, sequence, track = key
        validated_identifier(camera, "observation camera")
        validated_identifier(epoch, "observation epoch")
        validated_identifier(track, "observation local track ID")
        if (not isinstance(sequence, Integral) or isinstance(sequence, (bool, np.bool_))
                or not 0 <= sequence <= 2**63 - 1):
            raise ValueError("observation sequence must be a nonnegative signed 64-bit integer")
        normalized.append((camera, epoch, int(sequence), track))
    if len(set(normalized)) != len(normalized) or len({key[0] for key in normalized}) != len(normalized):
        raise ValueError("a fused person cannot reuse an observation or camera")
    return tuple(normalized)


@dataclass(frozen=True)
class FusedPerson:
    person_id: str
    identity_id: str | None
    identity_status: str
    world_points: np.ndarray
    valid_mask: np.ndarray
    joint_support: tuple
    reasons: tuple
    observation_keys: tuple

    def __post_init__(self):
        validated_identifier(self.person_id, "person_id")
        keys = _observation_keys(self.observation_keys)
        points = immutable_array(self.world_points, "world_points", allow_nan=True).copy()
        valid = np.array(self.valid_mask, copy=True)
        if (points.ndim != 2 or points.shape[1] != 3 or not 3 <= len(points) <= 256
                or valid.dtype.kind != "b" or valid.shape != (len(points),)):
            raise ValueError("world_points and valid_mask must have shapes (N,3) and (N,)")
        try:
            support = tuple(_identifiers(value, "joint support", maximum=8) for value in self.joint_support)
        except TypeError as exc:
            raise ValueError("joint_support must contain one camera-ID sequence per joint") from exc
        if len(support) != len(points) or np.any(valid & ~np.isfinite(points).all(axis=1)):
            raise ValueError("valid joints require finite coordinates and matching support")
        observed_cameras = {key[0] for key in keys}
        if any(set(cameras) - observed_cameras or (is_valid and len(cameras) < 2)
               for cameras, is_valid in zip(support, valid)):
            raise ValueError("valid joints require two distinct supporting observation cameras")
        if self.identity_status not in ("confirmed", "anonymous", "ambiguous"):
            raise ValueError("invalid identity status")
        if (self.identity_status == "confirmed") != (self.identity_id is not None):
            raise ValueError("only confirmed people may carry an identity")
        if self.identity_id is not None:
            validated_identifier(self.identity_id, "identity_id")
        reasons = _identifiers(self.reasons, "reasons", maximum=256)
        points[~valid] = np.nan
        points.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "world_points", points)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "joint_support", support)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "observation_keys", keys)

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "identity_id": self.identity_id,
            "identity_status": self.identity_status,
            "world_points_m": [point.tolist() if valid else None for point, valid in zip(self.world_points, self.valid_mask)],
            "valid_mask": self.valid_mask.tolist(),
            "joint_support": [list(s) for s in self.joint_support],
            "reasons": list(self.reasons),
            "observation_keys": [list(key) for key in self.observation_keys],
        }


@dataclass(frozen=True)
class FusedPacket:
    bundle_id: str
    timestamp_seconds: float
    persons: tuple
    missing_cameras: tuple = ()
    reasons: tuple = ()

    def __post_init__(self):
        validated_identifier(self.bundle_id, "bundle_id")
        timestamp = finite_scalar(self.timestamp_seconds, "timestamp_seconds", minimum=0)
        try:
            people = tuple(self.persons)
        except TypeError as exc:
            raise ValueError("persons must contain FusedPerson objects") from exc
        if len(people) > 64 or any(not isinstance(person, FusedPerson) for person in people):
            raise ValueError("packet must contain at most 64 FusedPerson objects")
        if len({person.person_id for person in people}) != len(people):
            raise ValueError("person IDs must be unique within a packet")
        identities = [person.identity_id for person in people if person.identity_id is not None]
        if len(set(identities)) != len(identities):
            raise ValueError("confirmed identity IDs must be unique within a packet")
        keys = [key for person in people for key in person.observation_keys]
        if len(set(keys)) != len(keys) or len(keys) > 64:
            raise ValueError("a packet cannot reuse observation provenance or exceed 64 observations")
        missing = _identifiers(self.missing_cameras, "missing_cameras", maximum=8)
        frame_provenance = defaultdict(set)
        for camera, epoch, sequence, _ in keys:
            frame_provenance[camera].add((epoch, sequence))
        if any(len(provenance) != 1 for provenance in frame_provenance.values()):
            raise ValueError("a packet may use only one frame epoch and sequence per camera")
        if len(set(frame_provenance) | set(missing)) > 8:
            raise ValueError("a packet may contain at most eight configured cameras")
        if set(missing) & set(frame_provenance):
            raise ValueError("observed cameras cannot also be missing")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "persons", people)
        object.__setattr__(self, "missing_cameras", missing)
        object.__setattr__(self, "reasons", _identifiers(self.reasons, "reasons", maximum=256))

    def to_dict(self):
        return {
            "schema_version": 1,
            "bundle_id": self.bundle_id,
            "timestamp_seconds": float(self.timestamp_seconds),
            "persons": [person.to_dict() for person in self.persons],
            "missing_cameras": list(self.missing_cameras),
            "reasons": list(self.reasons),
        }

    def to_json(self):
        """Canonical wire payload; absent geometry is JSON null, never NaN."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))


class MultiCameraFusion:
    """Fuse only geometrically consistent and unambiguous cross-camera people.

    Face confidence is supplied evidence, not a measured probability of identity.
    Confirmed labels gate association; geometry must independently support it.
    Anonymous identities are bundle-local and deliberately have no continuity.
    """

    def __init__(self, cameras, *, identity_threshold=0.9, max_span_seconds=0.04,
                 min_shared_joints=4, min_confidence=0.5, max_epipolar_error_px=3.0,
                 max_observations=64):
        values = list(cameras.values()) if hasattr(cameras, "values") else list(cameras)
        if any(not isinstance(camera, CalibratedCamera) for camera in values):
            raise ValueError("cameras must contain CalibratedCamera objects")
        self.cameras = MappingProxyType({camera.camera_id: camera for camera in values})
        if len(self.cameras) != len(values) or not 1 <= len(values) <= 8:
            raise ValueError("one to eight cameras must have distinct IDs")
        identity_threshold = finite_scalar(identity_threshold, "identity_threshold")
        if not 0 < identity_threshold <= 1:
            raise ValueError("identity_threshold must be in (0,1]")
        max_span_seconds = finite_scalar(max_span_seconds, "max_span_seconds", minimum=0)
        if isinstance(max_observations, bool) or not isinstance(max_observations, int) or not 1 <= max_observations <= 64:
            raise ValueError("max_observations must be an integer in [1,64]")
        if (isinstance(min_shared_joints, bool) or not isinstance(min_shared_joints, int)
                or not 3 <= min_shared_joints <= 256):
            raise ValueError("min_shared_joints must be an integer in [3,256]")
        min_confidence = finite_scalar(min_confidence, "min_confidence")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0,1]")
        max_epipolar_error_px = finite_scalar(max_epipolar_error_px, "max_epipolar_error_px")
        if max_epipolar_error_px <= 0:
            raise ValueError("max_epipolar_error_px must be positive and finite")
        self.identity_threshold = identity_threshold
        self.max_span_seconds = max_span_seconds
        self.max_observations = max_observations
        self.geometry_options = MappingProxyType(dict(min_shared_joints=min_shared_joints,
                                                      min_confidence=min_confidence,
                                                      max_epipolar_error_px=max_epipolar_error_px))

    def fuse(self, bundle):
        frames = tuple(sorted(bundle.frames, key=lambda frame: frame.camera_id))
        if len({frame.camera_id for frame in frames}) != len(frames):
            raise ValueError("a bundle may contain only one frame per camera")
        if any(frame.camera_id not in self.cameras for frame in frames):
            raise ValueError("bundle contains an uncalibrated camera")
        if any(camera_id not in self.cameras for camera_id in bundle.missing_cameras):
            raise ValueError("bundle lists an unknown missing camera")
        if {frame.camera_id for frame in frames} | set(bundle.missing_cameras) != set(self.cameras):
            raise ValueError("present and missing cameras must cover the configured rig")
        if len({frame.camera_id for frame in frames} & set(bundle.missing_cameras)):
            raise ValueError("a present camera cannot also be missing")
        timestamps = [frame.captured_at_seconds + self.cameras[frame.camera_id].clock_offset_seconds for frame in frames]
        # Revalidate timestamps at the boundary; a caller-created bundle cannot
        # bypass the synchronizer by declaring an incorrect span.
        if timestamps:
            uncertainties = [self.cameras[frame.camera_id].clock_uncertainty_seconds for frame in frames]
            lower = min(t - u for t, u in zip(timestamps, uncertainties))
            upper = max(t + u for t, u in zip(timestamps, uncertainties))
            if upper - lower > self.max_span_seconds + 1e-12:
                raise ValueError("bundle timestamps exceed the fusion synchronization window")
            if any(abs(t - bundle.target_timestamp_seconds) + u > self.max_span_seconds + 1e-12
                   for t, u in zip(timestamps, uncertainties)):
                raise ValueError("bundle target timestamp is inconsistent with its frames")
        records = []
        for frame in frames:
            local_ids = [observation.local_track_id for observation in frame.observations]
            if len(local_ids) != len(set(local_ids)):
                raise ValueError("duplicate local track IDs in one camera frame")
            for observation in sorted(frame.observations, key=lambda item: str(item.local_track_id)):
                key = (frame.camera_id, frame.stream_epoch, frame.sequence, observation.local_track_id)
                identity = observation.identity_id if observation.identity_confidence >= self.identity_threshold else None
                records.append((frame.camera_id, observation, key, identity))
        if len(records) > self.max_observations:
            raise ValueError("bundle exceeds the configured association workload limit")
        reasons = defaultdict(set)
        compatible = {}
        geometric = {}
        for i, (camera_a, first, _, id_a) in enumerate(records):
            for j in range(i + 1, len(records)):
                camera_b, second, _, id_b = records[j]
                if camera_a == camera_b:
                    compatible[i, j] = False
                    geometric[i, j] = False
                    continue
                check = estimate_pair_compatibility(first, second, self.cameras[camera_a],
                                                    self.cameras[camera_b], **self.geometry_options)
                geometric[i, j] = check.compatible
                compatible[i, j] = check.compatible and not (id_a is not None and id_b is not None and id_a != id_b)

        def matches(i, j):
            return i != j and compatible.get(tuple(sorted((i, j))), False)

        def geometric_matches(i, j):
            return i != j and geometric.get(tuple(sorted((i, j))), False)

        def has_alternative(i, j):
            return any(k != j and records[k][0] == records[j][0] and geometric_matches(i, k)
                       for k in range(len(records)))

        by_identity = defaultdict(list)
        for i, (_, _, _, identity) in enumerate(records):
            if identity is not None:
                by_identity[identity].append(i)
        groups = []
        assigned = set()
        blocked = set()
        for identity in sorted(by_identity):
            members = by_identity[identity]
            cameras = [records[i][0] for i in members]
            if len(cameras) != len(set(cameras)):
                reason = "duplicate_identity_in_camera"
            elif any(not matches(i, j) for k, i in enumerate(members) for j in members[k + 1:]):
                # With three or more views a uniquely supported majority may
                # survive one bad camera/label. At most eight cameras makes this
                # bounded exhaustive search cheap; tied subsets are ambiguous.
                subsets = []
                for size in range(len(members) - 1, 1, -1):
                    subsets = [subset for subset in combinations(members, size)
                               if all(matches(i, j) for i, j in combinations(subset, 2))]
                    if subsets:
                        break
                if len(subsets) == 1 and not any(has_alternative(i, j) or has_alternative(j, i)
                                                for i, j in combinations(subsets[0], 2)):
                    retained = list(subsets[0])
                    excluded = set(members) - set(retained)
                    groups.append((retained, identity))
                    assigned.update(retained)
                    for i in excluded:
                        reasons[i].add("quarantined_inconsistent_camera")
                        blocked.add(i)
                    continue
                reason = "identity_geometry_conflict"
            elif any(has_alternative(i, j) or has_alternative(j, i)
                     for k, i in enumerate(members) for j in members[k + 1:]):
                reason = "ambiguous_identity_geometry"
            else:
                groups.append((members.copy(), identity))
                assigned.update(members)
                continue
            for i in members:
                reasons[i].add(reason)
                blocked.add(i)

        # Attach a face-occluded/unknown observation only when exactly one known
        # group fits it, and no competing observation from its camera fits that
        # same group. Independently proposed attachments must agree pairwise.
        proposals = defaultdict(list)
        for i, (camera, _, _, identity) in enumerate(records):
            if i in assigned or i in blocked or identity is not None:
                continue
            candidates = [g for g, (members, _) in enumerate(groups)
                          if all(records[j][0] != camera and matches(i, j) for j in members)]
            if len(candidates) == 1:
                proposals[candidates[0]].append(i)
            elif len(candidates) > 1:
                reasons[i].add("ambiguous_identity_attachment")
                blocked.add(i)
        for g, proposed in proposals.items():
            by_camera = defaultdict(list)
            for i in proposed:
                by_camera[records[i][0]].append(i)
            conflicts = {i for members in by_camera.values() if len(members) > 1 for i in members}
            for n, i in enumerate(proposed):
                for j in proposed[n + 1:]:
                    if records[i][0] != records[j][0] and not matches(i, j):
                        conflicts.update((i, j))
            for i in proposed:
                if i in conflicts:
                    reasons[i].add("ambiguous_identity_attachment")
                    blocked.add(i)
                else:
                    groups[g][0].append(i)
                    assigned.add(i)

        # Anonymous association requires a unique partner per other camera and a
        # complete, mutually consistent component. Never use local tracker IDs to
        # bridge camera views or carry face labels across frames.
        remaining = set(range(len(records))) - assigned - blocked
        ambiguous = set()
        for i in remaining:
            neighbors = defaultdict(list)
            for j in remaining - {i}:
                if matches(i, j):
                    neighbors[records[j][0]].append(j)
            if any(len(members) > 1 for members in neighbors.values()):
                ambiguous.add(i)
                ambiguous.update(j for members in neighbors.values() if len(members) > 1 for j in members)
        for i in ambiguous:
            reasons[i].add("ambiguous_anonymous_association")
        blocked.update(ambiguous)
        remaining -= ambiguous
        while remaining:
            seed = min(remaining)
            component = {seed}
            pending = [seed]
            while pending:
                i = pending.pop()
                neighbors = {j for j in remaining - component if matches(i, j)}
                component.update(neighbors)
                pending.extend(neighbors)
            remaining -= component
            members = sorted(component)
            camera_ids = [records[i][0] for i in members]
            if len(set(camera_ids)) != len(camera_ids) or any(not matches(i, j) for n, i in enumerate(members) for j in members[n + 1:]):
                for i in members:
                    reasons[i].add("inconsistent_multiview_association")
                    blocked.add(i)
            else:
                groups.append((members, None))
                assigned.update(members)
        for i in sorted(blocked):
            groups.append(([i], None))
        people = []
        for members, identity in sorted(groups, key=lambda group: tuple(records[i][2] for i in sorted(group[0]))):
            members = sorted(members)
            observations = {records[i][0]: records[i][1] for i in members}
            estimates = fuse_pose(observations, self.cameras, min_confidence=self.geometry_options["min_confidence"])
            points = np.full((len(estimates), 3), np.nan)
            valid = np.zeros(len(estimates), dtype=bool)
            support = []
            group_reasons = {reason for i in members for reason in reasons[i]}
            for joint, estimate in enumerate(estimates):
                if estimate.position_m is not None:
                    points[joint] = estimate.position_m
                    valid[joint] = True
                else:
                    group_reasons.add(estimate.reason)
                support.append(estimate.supporting_cameras)
            keys = tuple(records[i][2] for i in members)
            token = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()[:20]
            status = "confirmed" if identity is not None else ("ambiguous" if any(reasons[i] for i in members) else "anonymous")
            people.append(FusedPerson(f"person-{token}", identity, status, points, valid,
                                      tuple(support), tuple(sorted(group_reasons)), keys))
        packet_reasons = ("incomplete_camera_bundle",) if bundle.missing_cameras else ()
        return FusedPacket(bundle.bundle_id, bundle.target_timestamp_seconds, tuple(people),
                           tuple(bundle.missing_cameras), packet_reasons)
