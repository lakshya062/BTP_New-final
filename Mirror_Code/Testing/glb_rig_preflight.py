#!/usr/bin/env python3
"""
Preflight validator for rigged GLB assets used in real-time avatar retargeting.

Checks:
  - glTF 2.0 container integrity
  - skin existence and mesh->skin binding
  - JOINTS_0 / WEIGHTS_0 attributes
  - joint index bounds and weight normalization
  - basic complexity metrics for edge-device budgets
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def parse_glb(path: Path) -> Tuple[Dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("File is too small to be a valid GLB.")

    magic, version, declared_len = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError("Invalid GLB magic.")
    if version != GLB_VERSION:
        raise ValueError(f"Unsupported GLB version: {version}")
    if declared_len != len(raw):
        raise ValueError("GLB declared length mismatch.")

    offset = 12
    gltf = None
    bin_chunk = b""
    while offset + 8 <= len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == JSON_CHUNK:
            gltf = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\0"))
        elif chunk_type == BIN_CHUNK:
            bin_chunk = payload

    if gltf is None:
        raise ValueError("No JSON chunk in GLB.")
    return gltf, bin_chunk


def read_accessor(gltf: Dict, bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPE[accessor["componentType"]]
    comps = TYPE_COMPONENTS[accessor["type"]]
    count = int(accessor["count"])
    acc_off = int(accessor.get("byteOffset", 0))
    view_off = int(view.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0))
    start = view_off + acc_off
    elem_nbytes = np.dtype(dtype).itemsize * comps

    if stride and stride != elem_nbytes:
        out = np.zeros((count, comps), dtype=dtype)
        for i in range(count):
            s = start + i * stride
            e = s + elem_nbytes
            out[i] = np.frombuffer(bin_chunk[s:e], dtype=dtype, count=comps)
        return out

    arr = np.frombuffer(
        bin_chunk,
        dtype=dtype,
        count=count * comps,
        offset=start,
    )
    return arr.reshape((count, comps))


def estimate_triangles(gltf: Dict, prim: Dict) -> int:
    mode = int(prim.get("mode", 4))
    if mode != 4:
        return 0
    idx_accessor = prim.get("indices")
    if idx_accessor is not None:
        return int(gltf["accessors"][idx_accessor]["count"]) // 3
    pos_accessor = prim["attributes"].get("POSITION")
    if pos_accessor is None:
        return 0
    return int(gltf["accessors"][pos_accessor]["count"]) // 3


def find_mesh_node(gltf: Dict, mesh_index: int) -> int:
    for ni, node in enumerate(gltf.get("nodes", [])):
        if node.get("mesh") == mesh_index:
            return ni
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rig readiness of a GLB asset.")
    parser.add_argument("--input", required=True, help="Input GLB path")
    parser.add_argument(
        "--report",
        default="",
        help="Optional output JSON report path",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    gltf, bin_chunk = parse_glb(in_path)

    errors: List[str] = []
    warnings: List[str] = []

    meshes = gltf.get("meshes", [])
    skins = gltf.get("skins", [])
    nodes = gltf.get("nodes", [])
    materials = gltf.get("materials", [])

    if not meshes:
        errors.append("No mesh found.")
        result = {"pass": False, "errors": errors, "warnings": warnings}
        print(json.dumps(result, indent=2))
        return

    mesh_index = 0
    prims = meshes[mesh_index].get("primitives", [])
    if not prims:
        errors.append("Primary mesh has no primitives.")
        result = {"pass": False, "errors": errors, "warnings": warnings}
        print(json.dumps(result, indent=2))
        return

    prim = prims[0]
    attrs = prim.get("attributes", {})
    mesh_node_index = find_mesh_node(gltf, mesh_index)
    skin_index = None if mesh_node_index < 0 else nodes[mesh_node_index].get("skin")

    if len(skins) == 0:
        errors.append("No skins found in GLB.")
    if mesh_node_index < 0:
        errors.append("No node references the primary mesh.")
    if mesh_node_index >= 0 and skin_index is None:
        errors.append("Mesh node is not bound to a skin.")

    for req_attr in ("POSITION", "JOINTS_0", "WEIGHTS_0"):
        if req_attr not in attrs:
            errors.append(f"Missing required vertex attribute: {req_attr}")

    position_count = None
    joints_count = None
    weights_count = None
    joint_min = None
    joint_max = None
    weight_sum_min = None
    weight_sum_max = None
    weight_nonzero_min = None

    if all(x in attrs for x in ("POSITION", "JOINTS_0", "WEIGHTS_0")):
        position = read_accessor(gltf, bin_chunk, attrs["POSITION"])
        joints = read_accessor(gltf, bin_chunk, attrs["JOINTS_0"])
        weights = read_accessor(gltf, bin_chunk, attrs["WEIGHTS_0"])
        position_count = int(position.shape[0])
        joints_count = int(joints.shape[0])
        weights_count = int(weights.shape[0])
        if position_count != joints_count or position_count != weights_count:
            errors.append(
                f"Vertex count mismatch POSITION={position_count}, JOINTS_0={joints_count}, "
                f"WEIGHTS_0={weights_count}"
            )

        if not np.isfinite(weights).all():
            errors.append("WEIGHTS_0 contains non-finite values.")
        if (weights < -1e-6).any():
            errors.append("WEIGHTS_0 contains negative values.")

        weight_sums = np.sum(weights, axis=1)
        weight_sum_min = float(weight_sums.min())
        weight_sum_max = float(weight_sums.max())
        if np.max(np.abs(weight_sums - 1.0)) > 0.02:
            warnings.append("Some vertices are not close to normalized weight sum 1.0 (+/-0.02).")

        nonzero = np.sum(weights > 1e-6, axis=1)
        weight_nonzero_min = int(nonzero.min())
        if weight_nonzero_min <= 0:
            errors.append("At least one vertex has zero effective skin weights.")

        joint_min = int(joints.min())
        joint_max = int(joints.max())
        if len(skins) > 0 and skin_index is not None:
            joint_count = len(skins[skin_index]["joints"])
            if joint_min < 0 or joint_max >= joint_count:
                errors.append(
                    f"JOINTS_0 index out of range [0,{joint_count - 1}] "
                    f"(observed min={joint_min}, max={joint_max})."
                )

    joint_count = 0
    has_ibm = False
    skeleton_node = None
    if len(skins) > 0 and skin_index is not None:
        skin = skins[skin_index]
        joint_count = len(skin.get("joints", []))
        has_ibm = "inverseBindMatrices" in skin
        skeleton_node = skin.get("skeleton")
        if joint_count < 15:
            warnings.append(f"Low joint count ({joint_count}); humanoid IK quality may be limited.")
        if not has_ibm:
            warnings.append("Skin has no inverseBindMatrices accessor.")

    triangles = estimate_triangles(gltf, prim)
    if triangles > 70000:
        warnings.append(
            f"Triangle count is high for edge use ({triangles}). "
            "Consider LOD/decimation for Orange Pi targets."
        )

    if len(materials) > 3:
        warnings.append(f"Material count is {len(materials)}; keep materials low for edge rendering.")

    result = {
        "asset": str(in_path),
        "pass": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "nodes": len(nodes),
            "meshes": len(meshes),
            "skins": len(skins),
            "materials": len(materials),
            "triangles_estimate": triangles,
            "mesh_node_index": mesh_node_index,
            "skin_index": skin_index,
            "joint_count": joint_count,
            "skeleton_node": skeleton_node,
            "has_inverse_bind_matrices": has_ibm,
            "vertex_count": position_count,
            "joints_count": joints_count,
            "weights_count": weights_count,
            "joint_index_min": joint_min,
            "joint_index_max": joint_max,
            "weight_sum_min": weight_sum_min,
            "weight_sum_max": weight_sum_max,
            "weight_nonzero_min": weight_nonzero_min,
        },
    }

    text = json.dumps(result, indent=2)
    print(text)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

