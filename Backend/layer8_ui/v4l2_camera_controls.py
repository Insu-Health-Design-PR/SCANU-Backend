"""UVC / V4L2 camera controls via ``v4l2-ctl`` (brightness, zoom, exposure, …)."""

from __future__ import annotations

import re
import subprocess
from typing import Any

_CTRL_HEAD = re.compile(
    r"^\s*(\S+)\s+0x[0-9a-fA-F]+\s+\((int|bool|menu|button|string|bitmask)\)\s*:\s*(.*)$"
)
_KV = re.compile(r"(\w+)=(-?\d+)")
_MENU_ITEM = re.compile(r"^\s+(\d+):\s*(.+)$")
_MENU_VALUE = re.compile(r"value=(\d+)\s*(?:\(([^)]+)\))?")


def device_path(index: int) -> str:
    return f"/dev/video{int(index)}"


def _run_v4l2_ctl(device: str, *args: str, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["v4l2-ctl", "-d", device, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_ctrl_attrs(tail: str) -> dict[str, Any]:
    out: dict[str, Any] = {"flags": []}
    for key, raw in _KV.findall(tail):
        out[key] = int(raw)
    flags_m = re.search(r"flags=([^\s]+(?:,[^\s]+)*)", tail)
    if flags_m:
        out["flags"] = [f.strip() for f in flags_m.group(1).split(",") if f.strip()]
    out["inactive"] = "inactive" in out.get("flags", [])
    menu_m = _MENU_VALUE.search(tail)
    if menu_m:
        out["value"] = int(menu_m.group(1))
        if menu_m.group(2):
            out["value_label"] = menu_m.group(2).strip()
    elif "value=" not in tail and re.search(r"value=(\d+)", tail) is None:
        bool_m = re.search(r"value=(\d+)", tail)
        if bool_m:
            out["value"] = int(bool_m.group(1))
    return out


def _parse_list_ctrls(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    pending_menu: dict[str, Any] | None = None

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("\t") and pending_menu is not None:
            mm = _MENU_ITEM.match(line)
            if mm:
                pending_menu.setdefault("menu_options", []).append(
                    {"value": int(mm.group(1)), "label": mm.group(2).strip()}
                )
            continue
        if not line.startswith((" ", "\t")) and line.endswith("Controls"):
            current_group = {"name": line.strip(), "controls": []}
            groups.append(current_group)
            pending_menu = None
            continue
        m = _CTRL_HEAD.match(line)
        if not m:
            continue
        cid, ctype, tail = m.group(1), m.group(2).lower(), m.group(3)
        attrs = _parse_ctrl_attrs(tail)
        ctrl: dict[str, Any] = {
            "id": cid,
            "type": ctype,
            "min": attrs.get("min"),
            "max": attrs.get("max"),
            "step": attrs.get("step", 1),
            "default": attrs.get("default"),
            "value": attrs.get("value"),
            "value_label": attrs.get("value_label"),
            "inactive": bool(attrs.get("inactive")),
            "flags": attrs.get("flags") or [],
            "menu_options": [],
        }
        if ctype == "bool" and ctrl["value"] is not None:
            ctrl["value"] = bool(int(ctrl["value"]))
        if ctype == "menu":
            pending_menu = ctrl
        else:
            pending_menu = None
        if current_group is not None:
            current_group["controls"].append(ctrl)
        else:
            groups.append({"name": "Controls", "controls": [ctrl]})

    flat: list[dict[str, Any]] = []
    for g in groups:
        flat.extend(g.get("controls") or [])
    return groups, flat


def list_camera_controls(index: int) -> dict[str, Any]:
    """Return UVC controls exposed on ``/dev/video{index}``."""
    dev = device_path(index)
    try:
        p = _run_v4l2_ctl(dev, "--list-ctrls-menus")
    except FileNotFoundError:
        return {"ok": False, "error": "v4l2-ctl not installed", "device": dev, "index": int(index)}
    except (subprocess.SubprocessError, OSError) as e:
        return {"ok": False, "error": str(e), "device": dev, "index": int(index)}

    text = p.stdout or ""
    if p.returncode != 0 and not text.strip():
        return {
            "ok": False,
            "error": (p.stderr or f"v4l2-ctl exit {p.returncode}").strip(),
            "device": dev,
            "index": int(index),
        }
    groups, flat = _parse_list_ctrls(text)
    return {
        "ok": True,
        "device": dev,
        "index": int(index),
        "groups": groups,
        "controls": flat,
        "raw": text,
    }


def set_camera_controls(index: int, values: dict[str, Any]) -> dict[str, Any]:
    """Apply one or more controls. Skips unknown keys and inactive controls."""
    dev = device_path(index)
    listed = list_camera_controls(index)
    if not listed.get("ok"):
        return listed
    by_id = {c["id"]: c for c in listed.get("controls") or []}
    parts: list[str] = []
    skipped: list[str] = []
    for key, raw_val in values.items():
        cid = str(key).strip()
        if not cid or cid not in by_id:
            skipped.append(cid)
            continue
        meta = by_id[cid]
        if meta.get("inactive"):
            skipped.append(cid)
            continue
        ctype = meta.get("type")
        if ctype == "bool":
            val = 1 if bool(raw_val) else 0
        elif ctype == "menu":
            val = int(raw_val)
        elif ctype == "int":
            val = int(raw_val)
        else:
            skipped.append(cid)
            continue
        parts.append(f"{cid}={val}")
    if not parts:
        return {
            "ok": False,
            "error": "no writable controls in request",
            "device": dev,
            "skipped": skipped,
        }
    arg = ",".join(parts)
    try:
        p = _run_v4l2_ctl(dev, f"--set-ctrl={arg}")
    except FileNotFoundError:
        return {"ok": False, "error": "v4l2-ctl not installed", "device": dev}
    except (subprocess.SubprocessError, OSError) as e:
        return {"ok": False, "error": str(e), "device": dev}
    if p.returncode != 0:
        return {
            "ok": False,
            "error": (p.stderr or p.stdout or f"v4l2-ctl exit {p.returncode}").strip(),
            "device": dev,
            "attempted": parts,
        }
    refreshed = list_camera_controls(index)
    return {
        "ok": True,
        "device": dev,
        "applied": parts,
        "skipped": skipped,
        "controls": refreshed.get("controls") or [],
    }


def reset_camera_controls(index: int) -> dict[str, Any]:
    """Reset writable controls to driver-reported defaults."""
    listed = list_camera_controls(index)
    if not listed.get("ok"):
        return listed
    to_set: dict[str, Any] = {}
    for c in listed.get("controls") or []:
        if c.get("inactive"):
            continue
        if c.get("default") is None:
            continue
        cid = str(c.get("id") or "")
        if not cid:
            continue
        if c.get("type") == "bool":
            to_set[cid] = bool(int(c["default"]))
        else:
            to_set[cid] = c["default"]
    return set_camera_controls(index, to_set)
