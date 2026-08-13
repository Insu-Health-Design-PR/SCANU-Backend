"""Detection helper modules."""

from weapon_ai.detection.firearms import (
    clamp_box,
    dedupe_gun_candidates,
    expand_person_roi_for_gun,
    firearm_kind_for_detection,
    is_armed_person_display_class,
    is_smartphone_display_class,
    person_armed_latch_class_allowed,
    gun_detection_valid,
    gun_yolo_binary_class_sets,
    nearest_person_ridx_for_gun,
    suppress_gun_phone_conflicts,
    xyxy_center_inside_any_person,
    yolo_keep_nonperson_detection,
)

__all__ = [
    "clamp_box",
    "dedupe_gun_candidates",
    "expand_person_roi_for_gun",
    "firearm_kind_for_detection",
    "gun_detection_valid",
    "gun_yolo_binary_class_sets",
    "nearest_person_ridx_for_gun",
    "suppress_gun_phone_conflicts",
    "xyxy_center_inside_any_person",
    "yolo_keep_nonperson_detection",
]
