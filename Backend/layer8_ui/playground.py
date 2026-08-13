"""Single-image weapon inference for the Layer 8 model playground."""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from layer8_ui.artifact_paths import software_root_from_settings
from layer8_ui.settings_store import load
from runtime.webcam_runner import _webcam_structured_weapon_args


def _merge_webcam_values(webcam: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    merged = {**webcam, **values}
    pm = merged.get("person_detection_model")
    if pm is not None and str(pm).strip():
        merged["weapon_yolo_model"] = str(pm).strip()
    return merged


def default_sample_image_path(layer8_dir: Path) -> Path:
    p = layer8_dir / "static" / "assets" / "playground" / "sample.jpg"
    if p.is_file():
        return p
    raise FileNotFoundError(f"Playground sample image missing: {p}")


def _software_env(sw: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{sw}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _infer_cwd(sw: Path) -> Path:
    """Run migrated ``weapon_ai`` from the backend root."""
    return sw


def build_playground_infer_cmd(
    *,
    layer8_dir: Path,
    source: Path,
    out_jpg: Path,
    out_json: Path,
    values: dict[str, Any] | None = None,
) -> list[str]:
    settings = load(layer8_dir)
    w = _merge_webcam_values(dict(settings.get("webcam") or {}), values or {})
    sw = software_root_from_settings(settings)
    py = os.environ.get("PYTHON", sys.executable)
    extra = _webcam_structured_weapon_args(w, sw).strip()
    yolo = str(w.get("person_detection_model") or w.get("weapon_yolo_model") or "").strip()
    cmd: list[str] = [
        py,
        "-m",
        "weapon_ai.infer_thermal_objects",
        "--source",
        str(source.resolve()),
        "--max_frames",
        "1",
        "--batch_warmup_passes",
        "0",
        "--no_imshow",
        "--playground_jpg",
        str(out_jpg.resolve()),
        "--playground_json",
        str(out_json.resolve()),
    ]
    if yolo:
        cmd.extend(["--yolo_model", yolo])
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def run_playground_infer(
    *,
    layer8_dir: Path,
    source: Path,
    values: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    settings = load(layer8_dir)
    sw = software_root_from_settings(settings)
    with tempfile.TemporaryDirectory(prefix="scanu_playground_") as tmp:
        out_jpg = Path(tmp) / "annotated.jpg"
        out_json = Path(tmp) / "metrics.json"
        cmd = build_playground_infer_cmd(
            layer8_dir=layer8_dir,
            source=source,
            out_jpg=out_jpg,
            out_json=out_json,
            values=values,
        )
        proc = subprocess.run(
            cmd,
            cwd=str(_infer_cwd(sw)),
            env=_software_env(sw),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
            raise RuntimeError(f"playground infer failed (exit {proc.returncode}): {tail}")
        if not out_jpg.is_file():
            raise RuntimeError("playground infer did not produce annotated JPEG")
        summary: dict[str, Any] = {}
        if out_json.is_file():
            import json

            summary = json.loads(out_json.read_text(encoding="utf-8"))
        image_b64 = base64.b64encode(out_jpg.read_bytes()).decode("ascii")
        return {
            "image_b64": image_b64,
            "image_mime": "image/jpeg",
            "summary": summary,
            "source": str(source.resolve()),
        }


def save_uploaded_image_b64(data_b64: str, *, suffix: str = ".jpg") -> Path:
    raw = base64.b64decode(data_b64, validate=True)
    fd, path = tempfile.mkstemp(prefix="scanu_playground_upload_", suffix=suffix)
    os.close(fd)
    p = Path(path)
    p.write_bytes(raw)
    return p
