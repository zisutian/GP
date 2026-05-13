from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image


_LMDB_ENV_CACHE = {}


def is_direct_grasp_record(data_item: dict[str, Any]) -> bool:
    return data_item.get("dataset") == "grasp_anything_direct"


def direct_grasp_conversation_preview(record: dict[str, Any]) -> list[dict[str, str]]:
    obj_name = record["obj_name"]
    return [
        {"from": "human", "value": f"<image>\ngrasp the {obj_name}"},
        {"from": "gpt", "value": "<loc0000><loc0000><loc0000><loc0000><loc0000>"},
    ]


def build_direct_grasp_item(record: dict[str, Any], image_size: int = 416, include_all_grasps: bool = False) -> dict[str, Any]:
    source_root = Path(record["source_root"])
    image = _load_image(source_root / "lmdb/image", record["image_key"])
    grasp_lmdb_path = source_root / "lmdb/grasp_label_positive"
    grasps = _load_grasps(grasp_lmdb_path, record["grasp_key"]) if include_all_grasps else None
    grasp = grasps[0] if grasps is not None else _load_grasp(grasp_lmdb_path, record["grasp_key"])
    obj_name = record["obj_name"]
    item = {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\ngrasp the {obj_name}"},
            {"from": "gpt", "value": _values_to_loc_tokens(_normalize_grasp(grasp, image_size))},
        ],
        "meta": record,
    }
    if grasps is not None:
        item["target_labels"] = grasps
    return item


def _get_lmdb_env(path: str | Path, max_readers: int = 2048):
    import lmdb

    lmdb_path = str(path)
    if lmdb_path not in _LMDB_ENV_CACHE:
        _LMDB_ENV_CACHE[lmdb_path] = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=max_readers,
        )
    return _LMDB_ENV_CACHE[lmdb_path]


def _get_lmdb_bytes(path: str | Path, key: str) -> bytes:
    env = _get_lmdb_env(path)
    with env.begin() as txn:
        value = txn.get(key.encode("utf-8"))
    if value is None:
        raise FileNotFoundError(f"LMDB key not found: {key} in {path}")
    return value


def _load_image(lmdb_path: str | Path, key: str) -> Image.Image:
    image_bytes = _get_lmdb_bytes(lmdb_path, key)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _load_grasp(lmdb_path: str | Path, key: str) -> list[float]:
    return _load_grasps(lmdb_path, key)[0]


def _load_grasps(lmdb_path: str | Path, key: str) -> list[list[float]]:
    import torch

    grasp_bytes = _get_lmdb_bytes(lmdb_path, key)
    buffer = io.BytesIO(grasp_bytes)
    try:
        grasp = torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        grasp = torch.load(buffer, map_location="cpu")
    return [[float(value) for value in row[1:]] for row in grasp]


def _normalize_grasp(grasp: list[float], image_size: int) -> list[float]:
    return [
        _clamp(grasp[0] / image_size),
        _clamp(grasp[1] / image_size),
        _clamp(grasp[2] / image_size),
        _clamp(grasp[3] / image_size),
        _clamp(grasp[4] / 180.0),
    ]


def _values_to_loc_tokens(values: list[float], bins: int = 1024) -> str:
    token_ids = [int(_clamp(value) * (bins - 1)) for value in values]
    return "".join(f"<loc{token_id:04d}>" for token_id in token_ids)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
