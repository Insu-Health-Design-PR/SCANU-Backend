"""v4l2-ctl helpers for the Layer 8 dashboard (auto-detect, format lists)."""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def _parse_list_devices_text(text: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current_name: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            current_name = None
            continue
        m = re.match(r"^/dev/(video\d+)", line.lstrip())
        if m and current_name is not None:
            idx = int(m.group(1).replace("video", ""))
            if groups and groups[-1].get("name") == current_name:
                if idx not in groups[-1]["indices"]:
                    groups[-1]["indices"].append(idx)
            else:
                groups.append(
                    {
                        "name": current_name,
                        "indices": [idx],
                        "nodes": [f"/dev/video{idx}"],
                    }
                )
            continue
        if line and not line.startswith(("\t", " ")):
            current_name = line.strip()
    for g in groups:
        g["indices"] = sorted(set(g["indices"]))
    return groups


def list_v4l2_groups() -> dict[str, Any]:
    text = ""
    v4l2_missing = False
    try:
        p = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = p.stdout or ""
        if p.returncode != 0 and not text.strip():
            return {
                "ok": False,
                "error": (p.stderr or f"v4l2-ctl exit {p.returncode}").strip(),
                "groups": _sysfs_video_groups() or _fallback_glob_video_groups(),
            }
    except FileNotFoundError:
        v4l2_missing = True
        groups = _sysfs_video_groups() or _fallback_glob_video_groups()
        return _build_v4l2_response(
            groups,
            "",
            extra_warning="v4l2-ctl not installed; enumerated cameras via sysfs.",
        )
    except (subprocess.SubprocessError, OSError) as e:
        groups = _sysfs_video_groups() or _fallback_glob_video_groups()
        if groups:
            return _build_v4l2_response(groups, "", extra_warning=str(e))
        return {"ok": False, "error": str(e), "groups": []}

    if not text.strip():
        groups = _sysfs_video_groups() or _fallback_glob_video_groups()
    else:
        groups = _parse_list_devices_text(text)
    if not groups:
        groups = _sysfs_video_groups() or _fallback_glob_video_groups()

    extra = "v4l2-ctl not installed; enumerated cameras via sysfs." if v4l2_missing else ""
    return _build_v4l2_response(groups, text, extra_warning=extra)


def _build_v4l2_response(
    groups: list[dict[str, Any]],
    raw: str,
    *,
    extra_warning: str = "",
) -> dict[str, Any]:
    usb_pat = re.compile(r"(?i)usb|uvc|webcam|camera|nexigo")
    thermal_pat = re.compile(r"(?i)purethermal|thermal|flir|seek|infrared")
    g_sorted = sorted(
        [g for g in groups if g.get("indices")],
        key=lambda g: min(g["indices"]),
    )
    suggested_webcam: int | None = None
    suggested_thermal: int | None = None
    warn = extra_warning.strip()
    thermal_groups = [g for g in g_sorted if thermal_pat.search(g.get("name", ""))]
    uvc_groups = [g for g in g_sorted if usb_pat.search(g.get("name", ""))]

    if thermal_groups:
        suggested_thermal = int(min(thermal_groups[0]["indices"]))
    if uvc_groups:
        suggested_webcam = int(min(uvc_groups[0]["indices"]))

    if len(g_sorted) >= 2:
        if suggested_thermal is None:
            non_uvc = [g for g in g_sorted if g not in uvc_groups] or g_sorted
            suggested_thermal = int(min(non_uvc[0]["indices"]))
        if suggested_webcam is None:
            suggested_webcam = int(min((uvc_groups or g_sorted)[-1]["indices"]))
    elif g_sorted:
        n = int(min(g_sorted[0]["indices"]))
        if suggested_thermal is None:
            suggested_thermal = n
        if suggested_webcam is None:
            suggested_webcam = n
        single_warn = "Single V4L2 group: thermal and webcam may use the same node; use two USB cameras to separate."
        warn = f"{warn} {single_warn}".strip() if warn else single_warn

    all_indices: list[int] = []
    for g in g_sorted:
        for i in g.get("indices", []):
            if int(i) not in all_indices:
                all_indices.append(int(i))
    all_indices.sort()
    if suggested_thermal is not None and suggested_webcam is not None and suggested_thermal == suggested_webcam and len(
        all_indices
    ) > 1:
        if thermal_groups and uvc_groups:
            suggested_thermal = int(min(thermal_groups[0]["indices"]))
            suggested_webcam = int(min(uvc_groups[0]["indices"]))
        else:
            suggested_thermal, suggested_webcam = all_indices[0], all_indices[-1]

    return {
        "ok": bool(g_sorted),
        "raw": raw,
        "groups": g_sorted,
        "all_indices": all_indices,
        "suggested_thermal": suggested_thermal,
        "suggested_webcam": suggested_webcam,
        "warning": warn,
    }


def _is_v4l_capture_node(index: int) -> bool:
    try:
        p = subprocess.run(
            ["udevadm", "info", "-q", "property", "-n", f"/dev/video{int(index)}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in (p.stdout or "").splitlines():
            if line.startswith("ID_V4L_CAPABILITIES="):
                return "capture" in line
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return True


def _sysfs_video_name(index: int) -> str:
    path = Path(f"/sys/class/video4linux/video{int(index)}/name")
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace").strip()
    return f"video{int(index)}"


def _sysfs_device_key(index: int) -> str:
    link = Path(f"/sys/class/video4linux/video{int(index)}")
    if link.is_symlink():
        target = os.path.realpath(link)
        if "/video4linux/" in target:
            return target.split("/video4linux/")[0]
    return f"video{int(index)}"


def _sysfs_video_groups() -> list[dict[str, Any]]:
    nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.sub(r"^\D+", "", Path(p).name) or 0),
    )
    if not nodes:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for node in nodes:
        try:
            index = int(re.sub(r"^\D+", "", Path(node).name) or -1)
        except ValueError:
            continue
        if index < 0 or not _is_v4l_capture_node(index):
            continue
        key = _sysfs_device_key(index)
        name = _sysfs_video_name(index)
        if key not in grouped:
            grouped[key] = {"name": name, "indices": [], "nodes": []}
        grouped[key]["indices"].append(index)
        grouped[key]["nodes"].append(node)

    out = list(grouped.values())
    for g in out:
        g["indices"] = sorted(set(g["indices"]))
        g["nodes"] = [f"/dev/video{i}" for i in g["indices"]]
    out.sort(key=lambda g: min(g["indices"]))
    return out


def _fallback_glob_video_groups() -> list[dict[str, Any]]:
    nodes = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.sub(r"^\D+", "", Path(p).name) or 0),
    )
    if not nodes:
        return []
    inds: list[int] = []
    for p in nodes:
        try:
            inds.append(int(re.sub(r"^\D+", "", Path(p).name) or 0))
        except ValueError:
            continue
    return [
        {
            "name": "Enumerated (no v4l2-ctl or empty list)",
            "indices": inds,
            "nodes": nodes,
        }
    ]


def list_formats_for_index(index: int) -> dict[str, Any]:
    path = f"/dev/video{int(index)}"
    if not Path(path).exists():
        return {
            "ok": False,
            "error": f"{path} not found",
            "options": _default_resolution_options(),
        }
    try:
        p = subprocess.run(
            ["v4l2-ctl", "-d", path, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=12,
        )
        out = p.stdout or ""
    except FileNotFoundError:
        return {"ok": False, "error": "v4l2-ctl not installed", "options": _default_resolution_options()}
    except (subprocess.SubprocessError, OSError) as e:
        return {"ok": False, "error": str(e), "options": _default_resolution_options()}

    options = _parse_formats_ext(out)
    if not options:
        options = _default_resolution_options()
    return {"ok": p.returncode == 0, "raw": out, "options": options}


def _parse_formats_ext(text: str) -> list[dict[str, Any]]:
    """Extract discrete WxH and fps from v4l2-ctl --list-formats-ext."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    w = h = 0
    for line in text.splitlines():
        m = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            continue
        m2 = re.search(r"\(([\d.]+) fps\)", line)
        if m2 and w > 0 and h > 0:
            fps = int(float(m2.group(1)))
            k = (w, h, fps)
            if k not in seen:
                seen.add(k)
                out.append(
                    {
                        "label": f"{w}×{h} @ {fps} fps",
                        "width": w,
                        "height": h,
                        "fps": float(fps),
                    }
                )
    out.sort(key=lambda o: (o["width"] * o["height"], o["fps"]), reverse=True)
    return out[:50]


def _default_resolution_options() -> list[dict[str, Any]]:
    """Fallback when v4l2-ctl is missing or returns nothing."""
    rows = [
        (3840, 2160, 30, "4K 3840×2160 @ 30"),
        (2560, 1440, 30, "1440p 2560×1440 @ 30"),
        (1920, 1080, 30, "1080p 1920×1080 @ 30"),
        (1280, 720, 30, "720p 1280×720 @ 30"),
        (640, 480, 30, "VGA 640×480 @ 30"),
    ]
    r = []
    for w, h, fps, label in rows:
        r.append({"label": label, "width": w, "height": h, "fps": float(fps)})
    return r


def list_serial_port_candidates() -> dict[str, Any]:
    """Prefer two ``ttyUSB*`` nodes (common for mmWave DCA) when present."""
    usb = sorted(glob.glob("/dev/ttyUSB*"), key=natural_path_sort)
    all_s = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"), key=natural_path_sort)
    if len(usb) >= 2:
        cli, data = usb[0], usb[1]
    elif len(all_s) >= 2:
        cli, data = all_s[0], all_s[1]
    elif all_s:
        cli, data = all_s[0], all_s[0]
    else:
        cli, data = "/dev/ttyUSB0", "/dev/ttyUSB1"
    return {"ok": True, "ports": all_s, "suggested_cli": cli, "suggested_data": data}


def natural_path_sort(p: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", p)
    return (int(m.group(1)) if m else 0, p)
