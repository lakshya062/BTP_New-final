"""Seeded multi-person detections and transport faults for internal rig tests.

Only this fixture knows the true actor behind an observation. Fusion receives
ordinary camera pixels, confidence, optional simulated identity evidence and
capture times. Occlusion is simulated by withholding landmarks, not by testing
a human pose detector against rendered people.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .camera_calibration import CameraCalibration
from .multicamera_types import CalibratedCamera, CameraFrame, PersonObservation
from .synthetic_camera import default_camera
from .synthetic_pose import generate_arm_pose, project_pose

SCENARIOS = (
    "clean_single", "complementary_occlusion", "one_view_joint", "camera_dropout",
    "jitter_reorder_duplicates", "mixed_frame_rates", "bad_joint_camera",
    "calibration_drift", "multiple_people", "crossing_ambiguity", "swapped_labels",
    "track_id_reuse", "anonymous_people", "uncertain_clock", "total_occlusion",
)


def make_rig(calibration=None):
    source = default_camera()
    calibration = calibration or CameraCalibration(source.camera_matrix, source.distortion, source.image_size)
    target = np.array([0., 0., 3.])
    centers = (("front", [0, 0, 0]), ("left", [-1.8, 0, .6]),
               ("right", [1.8, 0, .6]), ("high", [0, -1.0, .3]))
    offsets = (0., .180, -.120, .055)
    cameras = {}
    for (camera_id, center), offset in zip(centers, offsets):
        center = np.asarray(center, dtype=float)
        forward = target - center
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.vstack((right, down, forward))
        cameras[camera_id] = CalibratedCamera(camera_id, calibration, rotation, -rotation @ center,
                                               offset, .0005)
    return cameras


def body_pose(frame_index, actor, frame_count, *, crossing=False):
    """Return a known 33-joint metric body with animated left elbow/wrist."""
    phase = 2 * np.pi * frame_index / max(frame_count - 1, 1)
    angle = 100 + 60 * np.cos(phase + (0 if crossing else actor * .8))
    points = np.zeros((33, 3))
    for index in range(11):
        points[index] = [((index % 3) - 1) * .035, -.79 + (index // 3) * .015, -.06]
    points[0] = [0, -.83, -.075]
    points[7:11] = [[-.095, -.78, 0], [.095, -.78, 0], [-.04, -.74, -.065], [.04, -.74, -.065]]
    left = generate_arm_pose(angle, plane_yaw_degrees=35) + [-.23, -.18, 0]
    points[[11, 13, 15]] = left
    points[[12, 14, 16]] = [[.23, -.50, 0], [.32, -.18, -.02], [.26, .08, -.08]]
    for wrist, joints, sign in ((15, (17, 19, 21), -1), (16, (18, 20, 22), 1)):
        for n, joint in enumerate(joints):
            points[joint] = points[wrist] + [sign * (.02 + .008 * n), .025 + .012 * n, -.01 * n]
    points[23:33] = [[-.14, .06, 0], [.14, .06, 0], [-.14, .46, .04], [.14, .46, .04],
                     [-.14, .88, 0], [.14, .88, 0], [-.14, .91, .04], [.14, .91, .04],
                     [-.14, .92, -.14], [.14, .92, -.14]]
    scale = 1.0 if crossing else (1.0, .86, 1.10)[actor % 3]
    if crossing:
        progress = frame_index / max(frame_count - 1, 1)
        center = np.array([(-.48 + .96 * progress) * (1 if actor == 0 else -1), 0, 3.25])
    else:
        centers = ((-.60, -.05, 3.15), (.65, .15, 3.65), (.15, -.22, 4.05))
        center = np.asarray(centers[actor % 3])
    return points * scale + center


@dataclass
class SyntheticScene:
    scenario: str
    cameras: dict
    frames: tuple
    truth: dict
    actor_identities: dict
    frame_count: int


def make_scene(scenario, *, frame_count=18, seed=42, calibration=None, extrinsics=None, namespace="simulation"):
    if scenario not in SCENARIOS or isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 4:
        raise ValueError("known scenario and frame_count >= 4 required")
    truth_cameras = make_rig()
    cameras = make_rig(calibration)
    if extrinsics is not None:
        cameras = {key: replace(camera, rotation=extrinsics[key][0], translation_m=extrinsics[key][1])
                   for key, camera in cameras.items()}
    if scenario == "calibration_drift":
        cameras["front"] = replace(cameras["front"], translation_m=cameras["front"].translation_m + [0, .18, 0])
    if scenario == "uncertain_clock":
        cameras["front"] = replace(cameras["front"], clock_uncertainty_seconds=.030)
    rng = np.random.default_rng(seed)
    multi = scenario in {"multiple_people", "crossing_ambiguity", "swapped_labels", "anonymous_people", "jitter_reorder_duplicates"}
    identities = {actor: f"member-{actor}" for actor in range(2 if multi or scenario == "track_id_reuse" else 1)}
    frames, truth = [], {}
    epoch = f"{namespace}-{scenario}-{seed}"
    for frame_index in range(frame_count):
        actors = list(range(2)) if multi else [0]
        if scenario == "track_id_reuse":
            actors = [int(frame_index >= frame_count // 2)]
        for camera_index, (camera_id, camera) in enumerate(truth_cameras.items()):
            if scenario == "camera_dropout" and camera_id == "front" and frame_count // 3 <= frame_index < 2 * frame_count // 3:
                continue
            if scenario == "mixed_frame_rates" and ((camera_index == 1 and frame_index % 2)
                    or (camera_index == 2 and frame_index % 5 == 4) or (camera_index == 3 and frame_index % 3)):
                continue
            observations = []
            for actor in actors:
                crossing = scenario == "crossing_ambiguity"
                points = body_pose(frame_index, actor, frame_count, crossing=crossing)
                pixels = project_pose(points, camera.calibration.camera_matrix, camera.calibration.distortion_coefficients,
                                      cv2.Rodrigues(camera.rotation)[0], camera.translation_m)
                pixels += rng.normal(0, .20, pixels.shape)
                confidence = np.full(33, .97)
                hidden = set()
                if scenario == "complementary_occlusion":
                    hidden = ({13, 15, 17, 19, 21}, {14, 16, 18, 20, 22}, {23, 25, 27}, {24, 26, 28})[camera_index]
                elif scenario == "one_view_joint" and camera_id != "right":
                    hidden = {15}
                elif scenario == "total_occlusion":
                    hidden = set(range(33))
                hidden.update(np.flatnonzero(np.any((pixels < 0) | (pixels >= camera.calibration.image_size), axis=1)))
                if scenario == "bad_joint_camera" and camera_id == "front":
                    pixels[15] += [75, -55]
                if hidden:
                    indices = list(hidden)
                    pixels[indices] = np.nan
                    confidence[indices] = 0
                identity = identities[actor]
                if (scenario in {"anonymous_people", "crossing_ambiguity"}
                        or (scenario == "complementary_occlusion" and camera_id != "high")
                        or (scenario == "track_id_reuse" and actor == 1)):
                    identity = None
                if scenario == "swapped_labels" and camera_id == "front":
                    identity = identities[1 - actor]
                local_id = "recycled-track" if scenario == "track_id_reuse" else f"local-{camera_index}-{actor * 7 + 3}"
                observation = PersonObservation(local_id, pixels, confidence, identity, .99 if identity else 0)
                observations.append(observation)
                key = (camera_id, epoch, frame_index, local_id)
                truth[key] = (actor, frame_index, points)
            corrected = 1000 + frame_index / 30 + float(rng.uniform(-.0015, .0015))
            delay = float(rng.uniform(.005, .060)) if scenario == "jitter_reorder_duplicates" else .01 + camera_index * .001
            frame = CameraFrame(camera_id, frame_index, corrected - camera.clock_offset_seconds,
                                corrected + delay, tuple(observations), epoch)
            frames.append(frame)
            if scenario == "jitter_reorder_duplicates" and rng.random() < .25:
                frames.append(replace(frame, received_at_seconds=frame.received_at_seconds + .002))
    frames.sort(key=lambda frame: (frame.received_at_seconds, frame.camera_id, frame.sequence))
    return SyntheticScene(scenario, cameras, tuple(frames), truth, identities, frame_count)
