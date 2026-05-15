from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from data_tools.vcot_crop_lmdb import mask_to_bbox_position
from data_tools.vcot_direct_lmdb import (
    _clamp,
    _get_lmdb_bytes,
    _load_image,
    _values_to_loc_tokens,
)


def is_bbox_record(data_item: dict[str, Any]) -> bool:
    return data_item.get("dataset") == "grasp_anything_bbox"


def bbox_conversation_preview(record: dict[str, Any]) -> list[dict[str, str]]:
    obj_name = record["obj_name"]
    return [
        {"from": "human", "value": f"<image>\ndetect {obj_name}"},
        {"from": "gpt", "value": "<loc0000><loc0000><loc0000><loc0000>"},
    ]


def build_bbox_item(record: dict[str, Any], image_size: int = 416) -> dict[str, Any]:
    source_root = Path(record["source_root"])
    image = _load_image(source_root / "lmdb/image", record["image_key"])
    mask_key = record["mask_key"]
    mask = _load_mask(source_root / "lmdb/mask", mask_key)
    bbox = mask_to_bbox_position(mask)
    if bbox is None:
        raise ValueError(f"Empty mask for record: {record.get('grasp_id')}")

    obj_name = record["obj_name"]
    target_norm = normalize_bbox_xyxy(bbox, image_size)
    meta = dict(record)
    meta.update({
        "bbox": bbox,
        "target_coordinate_frame": "full_image",
        "target_bbox_order": "xyxy",
    })
    return {
        "image": image,
        "conversations": [
            {"from": "human", "value": f"<image>\ndetect {obj_name}"},
            {"from": "gpt", "value": _values_to_loc_tokens(target_norm)},
        ],
        "meta": meta,
        "target_bbox": bbox,
        "target_norm": target_norm,
    }


def normalize_bbox_xyxy(bbox: list[int] | tuple[int, int, int, int], image_size: int = 416) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [
        _clamp(x0 / image_size),
        _clamp(y0 / image_size),
        _clamp(x1 / image_size),
        _clamp(y1 / image_size),
    ]


def denormalize_bbox_xyxy(values: list[float], image_size: int = 416) -> list[int]:
    return [int(_clamp(value) * image_size) for value in values[:4]]


def _load_mask(lmdb_path: str | Path, key: str) -> np.ndarray:
    mask_bytes = _get_lmdb_bytes(lmdb_path, key)
    return np.load(io.BytesIO(mask_bytes))
