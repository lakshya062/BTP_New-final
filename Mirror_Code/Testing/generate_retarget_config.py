#!/usr/bin/env python3
"""
Generate a runtime retarget configuration from a rigged GLB.

Outputs a JSON config with:
  - canonical humanoid bone slots
  - IK chains
  - detector-keypoint -> rig-joint hints
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A


def parse_glb_json(path: Path) -> Dict:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("File too small for GLB.")
    magic, version, declared_len = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError("Invalid GLB magic.")
    if version != GLB_VERSION:
        raise ValueError(f"Unsupported GLB version: {version}")
    if declared_len != len(raw):
        raise ValueError("GLB declared length mismatch.")

    offset = 12
    while offset + 8 <= len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        data = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == JSON_CHUNK:
            return json.loads(data.decode("utf-8").rstrip(" \t\r\n\0"))
    raise ValueError("No JSON chunk found.")


def node_name_list(gltf: Dict, joint_indices: List[int]) -> Dict[str, int]:
    nodes = gltf.get("nodes", [])
    name_to_index = {}
    for ji in joint_indices:
        if ji < 0 or ji >= len(nodes):
            continue
        name = str(nodes[ji].get("name", f"joint_{ji}"))
        name_to_index[name] = ji
    return name_to_index


def find_by_keywords(names: Dict[str, int], keywords: List[str]) -> str | None:
    lowered = {k.lower(): k for k in names.keys()}
    for kw in keywords:
        for lk, orig in lowered.items():
            if kw in lk:
                return orig
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retarget config from GLB skin joints.")
    parser.add_argument("--input", required=True, help="Input rigged GLB path")
    parser.add_argument("--output", required=True, help="Output retarget JSON path")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    gltf = parse_glb_json(in_path)
    skins = gltf.get("skins", [])
    nodes = gltf.get("nodes", [])
    if not skins:
        raise RuntimeError("No skins found in GLB. Cannot generate retarget config.")

    skin_idx = 0
    joint_indices = list(skins[skin_idx].get("joints", []))
    names = node_name_list(gltf, joint_indices)

    slots = {
        "root": find_by_keywords(names, ["hips", "pelvis", "root"]),
        "spine": find_by_keywords(names, ["spine"]),
        "chest": find_by_keywords(names, ["chest", "upperchest"]),
        "neck": find_by_keywords(names, ["neck"]),
        "head": find_by_keywords(names, ["head"]),
        "left_shoulder": find_by_keywords(names, ["leftshoulder", "l_shoulder", "shoulder_l"]),
        "left_upper_arm": find_by_keywords(names, ["leftupperarm", "l_upperarm", "upperarm_l"]),
        "left_lower_arm": find_by_keywords(names, ["leftlowerarm", "l_forearm", "lowerarm_l"]),
        "left_hand": find_by_keywords(names, ["lefthand", "l_hand", "hand_l"]),
        "right_shoulder": find_by_keywords(names, ["rightshoulder", "r_shoulder", "shoulder_r"]),
        "right_upper_arm": find_by_keywords(names, ["rightupperarm", "r_upperarm", "upperarm_r"]),
        "right_lower_arm": find_by_keywords(names, ["rightlowerarm", "r_forearm", "lowerarm_r"]),
        "right_hand": find_by_keywords(names, ["righthand", "r_hand", "hand_r"]),
        "left_upper_leg": find_by_keywords(names, ["leftupperleg", "l_thigh", "upperleg_l"]),
        "left_lower_leg": find_by_keywords(names, ["leftlowerleg", "l_calf", "lowerleg_l"]),
        "left_foot": find_by_keywords(names, ["leftfoot", "l_foot", "foot_l"]),
        "right_upper_leg": find_by_keywords(names, ["rightupperleg", "r_thigh", "upperleg_r"]),
        "right_lower_leg": find_by_keywords(names, ["rightlowerleg", "r_calf", "lowerleg_r"]),
        "right_foot": find_by_keywords(names, ["rightfoot", "r_foot", "foot_r"]),
    }

    missing_slots = [k for k, v in slots.items() if v is None]

    output = {
        "asset_glb": str(in_path),
        "skin_index": skin_idx,
        "joint_count": len(joint_indices),
        "skeleton_node": skins[skin_idx].get("skeleton"),
        "bone_slots": slots,
        "ik_chains": {
            "left_arm": {
                "root": slots["left_upper_arm"],
                "mid": slots["left_lower_arm"],
                "end": slots["left_hand"],
                "pole_hint": "left_elbow_out",
            },
            "right_arm": {
                "root": slots["right_upper_arm"],
                "mid": slots["right_lower_arm"],
                "end": slots["right_hand"],
                "pole_hint": "right_elbow_out",
            },
            "left_leg": {
                "root": slots["left_upper_leg"],
                "mid": slots["left_lower_leg"],
                "end": slots["left_foot"],
                "pole_hint": "left_knee_forward",
            },
            "right_leg": {
                "root": slots["right_upper_leg"],
                "mid": slots["right_lower_leg"],
                "end": slots["right_foot"],
                "pole_hint": "right_knee_forward",
            },
        },
        "detector_keypoint_targets": {
            # COCO-17 / YOLO-style indices
            "5_left_shoulder": slots["left_shoulder"] or slots["left_upper_arm"],
            "6_right_shoulder": slots["right_shoulder"] or slots["right_upper_arm"],
            "7_left_elbow": slots["left_lower_arm"],
            "8_right_elbow": slots["right_lower_arm"],
            "9_left_wrist": slots["left_hand"],
            "10_right_wrist": slots["right_hand"],
            "11_left_hip": slots["left_upper_leg"] or slots["root"],
            "12_right_hip": slots["right_upper_leg"] or slots["root"],
            "13_left_knee": slots["left_lower_leg"],
            "14_right_knee": slots["right_lower_leg"],
            "15_left_ankle": slots["left_foot"],
            "16_right_ankle": slots["right_foot"],
            "0_nose": slots["head"],
        },
        "all_skin_joints": [
            {"index": ji, "name": nodes[ji].get("name", f"joint_{ji}")}
            for ji in joint_indices
            if 0 <= ji < len(nodes)
        ],
        "warnings": [],
    }

    if missing_slots:
        output["warnings"].append(
            "Some canonical bone slots were not resolved: " + ", ".join(missing_slots)
        )

    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Retarget config written: {out_path}")
    if output["warnings"]:
        print("Warnings:")
        for w in output["warnings"]:
            print(f"- {w}")


if __name__ == "__main__":
    main()

