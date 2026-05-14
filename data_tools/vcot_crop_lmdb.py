from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from data_tools.vcot_direct_lmdb import (
    _clamp,
    _get_lmdb_bytes,
    _load_grasps,
    _load_image,
    _normalize_grasp,
    _values_to_loc_tokens,
)


DEFAULT_BBOX_EDGE_EXPAND = 15
DEFAULT_MIN_BBOX_HALF_SIZE = 50
TARGET_FRAME_FULL_IMAGE = "full_image"
TARGET_FRAME_CROP_IMAGE = "crop_image"


def is_crop_grasp_record(data_item: dict[str, Any]) -> bool:
    return data_item.get("dataset") == "grasp_anything_crop"


def crop_grasp_conversation_preview(record: dict[str, Any]) -> list[dict[str, str]]:
    obj_name = record["obj_name"]
    return [
        {"from": "human", "value": f"<image>\n<image>\ngrasp the {obj_name}"},
        {"from": "gpt", "value": "<loc0000><loc0000><loc0000><loc0000><loc0000>"},
    ]


def build_crop_grasp_item(
    record: dict[str, Any],
    image_size: int = 416,
    bbox_edge_expand: int | None = None,
    min_bbox_half_size: int | None = None,
    target_coordinate_frame: str | None = None,
    include_all_grasps: bool = False,
) -> dict[str, Any]:
    source_root = Path(record["source_root"])
    full_image = _load_image(source_root / "lmdb/image", record["image_key"])

    grasp_lmdb_path = source_root / "lmdb/grasp_label_positive"
    grasps = _load_grasps(grasp_lmdb_path, record["grasp_key"])
    target_index = _target_grasp_index(record, len(grasps))
    target_grasp = grasps[target_index]

    mask_key = record.get("mask_key", f"{record['grasp_id']}.npy")
    mask = _load_mask(source_root / "lmdb/mask", mask_key)
    object_bbox = mask_to_bbox_position(mask)
    if object_bbox is None:
        raise ValueError(f"Empty mask for record: {record.get('grasp_id')}")

    bbox_edge_expand = int(
        bbox_edge_expand if bbox_edge_expand is not None else record.get("bbox_edge_expand", DEFAULT_BBOX_EDGE_EXPAND)
    )
    min_bbox_half_size = int(
        min_bbox_half_size
        if min_bbox_half_size is not None
        else record.get("min_bbox_half_size", DEFAULT_MIN_BBOX_HALF_SIZE)
    )
    crop_box = crop_box_from_bbox(
        object_bbox,
        image_size=image_size,
        min_half_size=min_bbox_half_size,
        edge_expand=bbox_edge_expand,
    )
    pil_crop_box = _scale_box_for_image(crop_box, label_image_size=image_size, image_size=full_image.size)
    crop_image = full_image.crop(tuple(pil_crop_box))
    target_coordinate_frame = (
        target_coordinate_frame
        if target_coordinate_frame is not None
        else record.get("target_coordinate_frame", TARGET_FRAME_FULL_IMAGE)
    )
    if target_coordinate_frame == TARGET_FRAME_FULL_IMAGE:
        target_norm = _normalize_grasp(target_grasp, image_size)
    elif target_coordinate_frame == TARGET_FRAME_CROP_IMAGE:
        target_norm = transform_grasp_to_crop_norm(target_grasp, crop_box)
    else:
        raise ValueError(f"Unsupported crop target coordinate frame: {target_coordinate_frame}")
    obj_name = record["obj_name"]

    meta = dict(record)
    meta.update({
        "crop_source": "gt_mask_object_bbox",
        "object_bbox": object_bbox,
        "crop_box": crop_box,
        "pil_crop_box": pil_crop_box,
        "bbox_edge_expand": bbox_edge_expand,
        "min_bbox_half_size": min_bbox_half_size,
        "target_grasp_index": target_index,
        "target_grasp": target_grasp,
        "target_coordinate_frame": target_coordinate_frame,
    })
    item = {
        "image": [full_image, crop_image],
        "conversations": [
            {"from": "human", "value": f"<image>\n<image>\ngrasp the {obj_name}"},
            {"from": "gpt", "value": _values_to_loc_tokens(target_norm)},
        ],
        "meta": meta,
        "crop_box": crop_box,
        "object_bbox": object_bbox,
        "target_grasp": target_grasp,
        "target_norm": target_norm,
        "target_coordinate_frame": target_coordinate_frame,
    }
    if include_all_grasps:
        item["target_labels"] = grasps
    return item


def mask_to_bbox_position(mask: np.ndarray) -> list[int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def transform_grasp_to_crop_norm(
    grasp: list[float] | tuple[float, float, float, float, float],
    crop_box: list[int] | tuple[int, int, int, int],
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    x, y, w, h, angle = [float(value) for value in grasp[:5]]
    return [
        _clamp((x - x0) / crop_w),
        _clamp((y - y0) / crop_h),
        _clamp(w / crop_w),
        _clamp(h / crop_h),
        _clamp(angle / 180.0),
    ]


def transform_grasp_from_crop_norm(
    values: list[float] | tuple[float, float, float, float, float],
    crop_box: list[int] | tuple[int, int, int, int],
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    x, y, w, h, angle = [_clamp(value) for value in values[:5]]
    return [
        x0 + x * crop_w,
        y0 + y * crop_h,
        w * crop_w,
        h * crop_h,
        angle * 180.0,
    ]


def crop_box_from_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    image_size: int | tuple[int, int] = 416,
    min_half_size: int = DEFAULT_MIN_BBOX_HALF_SIZE,
    edge_expand: int = DEFAULT_BBOX_EDGE_EXPAND,
) -> list[int]:
    """VCoT-style visual sampler: square crop around an object bbox."""
    image_width, image_height = _image_extent(image_size)
    x_min, y_min, x_max, y_max = [float(value) for value in bbox]

    if sum([x_min, y_min, x_max, y_max]) < 5:
        scale = max(image_width, image_height)
        x_min *= scale
        y_min *= scale
        x_max *= scale
        y_max *= scale

    if image_width > image_height:
        overlay = (image_width - image_height) // 2
        y_min = max(0, y_min - overlay)
        y_max = max(0, y_max - overlay)
    else:
        overlay = (image_height - image_width) // 2
        x_min = max(0, x_min - overlay)
        x_max = max(0, x_max - overlay)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    half_w = (x_max - x_min) // 2
    half_h = (y_max - y_min) // 2
    half_size = max(max(half_w, half_h) + edge_expand, min_half_size)

    if center_x - half_size < 0:
        center_x += -(center_x - half_size)
    if center_y - half_size < 0:
        center_y += -(center_y - half_size)
    if center_x + half_size > image_width:
        center_x -= center_x + half_size - image_width
    if center_y + half_size > image_height:
        center_y -= center_y + half_size - image_height

    return [
        max(0, int(center_x - half_size)),
        max(0, int(center_y - half_size)),
        min(image_width, int(center_x + half_size)),
        min(image_height, int(center_y + half_size)),
    ]


def _load_mask(lmdb_path: str | Path, key: str) -> np.ndarray:
    mask_bytes = _get_lmdb_bytes(lmdb_path, key)
    return np.load(io.BytesIO(mask_bytes))


def _target_grasp_index(record: dict[str, Any], num_grasps: int) -> int:
    if num_grasps <= 0:
        raise ValueError(f"No positive grasp labels for record: {record.get('grasp_id')}")
    index = int(record.get("target_grasp_index", 0))
    return max(0, min(num_grasps - 1, index))


def _image_extent(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, tuple):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def _scale_box_for_image(crop_box: list[int], label_image_size: int, image_size: tuple[int, int]) -> list[int]:
    image_width, image_height = int(image_size[0]), int(image_size[1])
    if image_width == label_image_size and image_height == label_image_size:
        return [int(value) for value in crop_box]

    x_scale = image_width / float(label_image_size)
    y_scale = image_height / float(label_image_size)
    x0, y0, x1, y1 = crop_box
    return [
        max(0, min(image_width, math.floor(x0 * x_scale))),
        max(0, min(image_height, math.floor(y0 * y_scale))),
        max(0, min(image_width, math.ceil(x1 * x_scale))),
        max(0, min(image_height, math.ceil(y1 * y_scale))),
    ]
