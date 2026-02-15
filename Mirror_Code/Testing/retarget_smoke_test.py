#!/usr/bin/env python3
"""
Retarget smoke test:
  - loads generated retarget config
  - builds synthetic COCO-17 keypoints
  - computes basic bone quaternions for key chains
  - writes result JSON

This is a numeric preflight only; it does not render.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = normalize(v_from)
    b = normalize(v_to)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -0.999999:
        # 180-degree fallback around an arbitrary perpendicular axis.
        axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
        if np.linalg.norm(axis) < 1e-6:
            axis = normalize(np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)
    axis = normalize(np.cross(a, b))
    s = np.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / max(s, 1e-8)
    q = np.array(
        [
            0.5 * s,
            axis[0] * inv_s,
            axis[1] * inv_s,
            axis[2] * inv_s,
        ],
        dtype=np.float32,
    )
    return q / max(np.linalg.norm(q), 1e-8)


def synthetic_coco17() -> np.ndarray:
    # Normalized synthetic pose in camera space (x right, y down, z forward=0).
    pts = np.zeros((17, 3), dtype=np.float32)
    pts[0] = [0.50, 0.16, 0.0]  # nose
    pts[5] = [0.43, 0.28, 0.0]  # left shoulder
    pts[6] = [0.57, 0.28, 0.0]  # right shoulder
    pts[7] = [0.37, 0.39, 0.0]  # left elbow
    pts[8] = [0.63, 0.39, 0.0]  # right elbow
    pts[9] = [0.33, 0.51, 0.0]  # left wrist
    pts[10] = [0.67, 0.51, 0.0]  # right wrist
    pts[11] = [0.46, 0.50, 0.0]  # left hip
    pts[12] = [0.54, 0.50, 0.0]  # right hip
    pts[13] = [0.45, 0.69, 0.0]  # left knee
    pts[14] = [0.55, 0.69, 0.0]  # right knee
    pts[15] = [0.44, 0.88, 0.03]  # left ankle
    pts[16] = [0.56, 0.88, 0.03]  # right ankle
    # Eyes/ears not used here.
    return pts


def required_chain_slots(config: Dict) -> Dict[str, Tuple[str, str, str]]:
    slots = config["bone_slots"]
    return {
        "left_upper_arm": ("5_left_shoulder", "7_left_elbow", slots["left_upper_arm"]),
        "left_lower_arm": ("7_left_elbow", "9_left_wrist", slots["left_lower_arm"]),
        "right_upper_arm": ("6_right_shoulder", "8_right_elbow", slots["right_upper_arm"]),
        "right_lower_arm": ("8_right_elbow", "10_right_wrist", slots["right_lower_arm"]),
        "left_upper_leg": ("11_left_hip", "13_left_knee", slots["left_upper_leg"]),
        "left_lower_leg": ("13_left_knee", "15_left_ankle", slots["left_lower_leg"]),
        "right_upper_leg": ("12_right_hip", "14_right_knee", slots["right_upper_leg"]),
        "right_lower_leg": ("14_right_knee", "16_right_ankle", slots["right_lower_leg"]),
    }


def key_id_to_index(token: str) -> int:
    # token format: "7_left_elbow"
    return int(token.split("_", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget smoke test (no rendering).")
    parser.add_argument("--config", required=True, help="Path to retarget JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    missing = []
    for k, v in cfg.get("bone_slots", {}).items():
        if v is None:
            missing.append(k)
    if missing:
        raise RuntimeError("Retarget config missing required slots: " + ", ".join(missing))

    pose = synthetic_coco17()
    chains = required_chain_slots(cfg)
    default_bone_forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    rotations = {}
    for chain_name, (from_token, to_token, bone_name) in chains.items():
        i0 = key_id_to_index(from_token)
        i1 = key_id_to_index(to_token)
        direction = normalize(pose[i1] - pose[i0])
        q = quat_from_two_vectors(default_bone_forward, direction)
        rotations[chain_name] = {
            "bone": bone_name,
            "from_keypoint": from_token,
            "to_keypoint": to_token,
            "direction_xyz": [float(x) for x in direction],
            "rotation_quat_wxyz": [float(x) for x in q],
        }

    output = {
        "asset": cfg.get("asset_glb"),
        "config": str(cfg_path),
        "status": "ok",
        "note": "Smoke-test quaternions from synthetic pose. Use as integration sanity check only.",
        "rotations": rotations,
    }
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Smoke test output written: {out_path}")


if __name__ == "__main__":
    main()

