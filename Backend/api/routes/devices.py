"""Device discovery routes."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from legacy_layer8.adapters import ensure_legacy_imports

ensure_legacy_imports()

from layer8_ui import v4l2_tools  # noqa: E402
from layer8_ui import v4l2_camera_controls  # noqa: E402


class V4l2SetControlsBody(BaseModel):
    index: int = Field(..., ge=0, le=64, description="/dev/videoN index")
    controls: dict[str, Any] = Field(..., description="Control id → value, e.g. brightness: 128")


class ProbeSourceBody(BaseModel):
    source: str = Field(..., description="Local index (0), /dev/videoN, or rtsp://… / http://… URL")


def _probe_video_source(source: str, timeout_s: float | None = None) -> dict[str, Any]:
    """Open a V4L2 node or network URL briefly and report OK/Error."""
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    import threading

    raw = str(source or "").strip()
    if not raw:
        return {"ok": False, "error": "empty source", "source": raw}

    opened_as = raw
    if re.fullmatch(r"\d+", raw):
        opened_as = f"/dev/video{int(raw)}"
    elif re.fullmatch(r"/dev/video\d+", raw):
        opened_as = raw

    is_network = bool(re.match(r"^(rtsp|rtsps|http|https|udp|tcp)://", opened_as, re.I))
    if timeout_s is None:
        timeout_s = 12.0 if is_network else 5.0
    if not is_network and opened_as.startswith("/dev/video"):
        if not os.path.exists(opened_as):
            return {"ok": False, "error": f"{opened_as} does not exist", "source": opened_as}
        try:
            from layer8_ui.webcam_device import is_video_capture_node

            idx_m = re.search(r"/dev/video(\d+)$", opened_as)
            if idx_m and not is_video_capture_node(int(idx_m.group(1))):
                return {
                    "ok": False,
                    "error": f"{opened_as} is a metadata node (not Video Capture)",
                    "source": opened_as,
                    "metadata": True,
                }
        except Exception:
            pass

    result: dict[str, Any] = {}

    def _worker() -> None:
        import cv2

        t0 = time.time()
        cap = None
        try:
            if is_network:
                cap = cv2.VideoCapture(opened_as, cv2.CAP_FFMPEG)
            else:
                idx_m = re.search(r"/dev/video(\d+)$", opened_as)
                if idx_m:
                    cap = cv2.VideoCapture(int(idx_m.group(1)), cv2.CAP_V4L2)
                    if cap is None or not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        cap = cv2.VideoCapture(opened_as)
                else:
                    cap = cv2.VideoCapture(opened_as)
            if cap is None or not cap.isOpened():
                result.update(
                    {
                        "ok": False,
                        "error": f"Cannot open {opened_as}",
                        "source": opened_as,
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )
                return
            ok, frame = False, None
            for _ in range(12):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    break
                time.sleep(0.05)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if ok and frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
                h = int(frame.shape[0])
                w = int(frame.shape[1])
            if not ok or frame is None:
                # HDMI H.264 capture nodes often fail OpenCV read but work with FFmpeg CUVID.
                try:
                    from layer8_ui.webcam_device import is_video_capture_node

                    idx_m = re.search(r"/dev/video(\d+)$", opened_as)
                    if idx_m and is_video_capture_node(int(idx_m.group(1))):
                        result.update(
                            {
                                "ok": True,
                                "message": "OK (capture node; H.264 — OpenCV may not decode; FFmpeg CUVID will)",
                                "source": opened_as,
                                "width": w,
                                "height": h,
                                "fps": round(fps, 2) if fps else None,
                                "elapsed_s": round(time.time() - t0, 3),
                                "network": False,
                                "hdmi_h264": True,
                            }
                        )
                        return
                except Exception:
                    pass
                result.update(
                    {
                        "ok": False,
                        "error": f"Opened {opened_as} but failed to read a frame",
                        "source": opened_as,
                        "width": w,
                        "height": h,
                        "fps": fps,
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )
                return
            result.update(
                {
                    "ok": True,
                    "message": "OK",
                    "source": opened_as,
                    "width": w,
                    "height": h,
                    "fps": round(fps, 2) if fps else None,
                    "elapsed_s": round(time.time() - t0, 3),
                    "network": is_network,
                }
            )
        except Exception as e:
            result.update(
                {
                    "ok": False,
                    "error": str(e),
                    "source": opened_as,
                    "elapsed_s": round(time.time() - t0, 3),
                }
            )
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=float(timeout_s))
    if th.is_alive():
        return {
            "ok": False,
            "error": f"Probe timed out after {timeout_s:.0f}s opening {opened_as}",
            "source": opened_as,
            "elapsed_s": float(timeout_s),
        }
    return result or {"ok": False, "error": "probe produced no result", "source": opened_as}

def build_devices_router(_ctx: object) -> APIRouter:
    router = APIRouter(tags=["devices"])

    @router.get("/api/devices/v4l2")
    def v4l2_devices() -> dict[str, Any]:
        """`v4l2-ctl --list-devices` with suggested indices for thermal vs webcam."""
        return v4l2_tools.list_v4l2_groups()

    @router.get("/api/devices/v4l2/formats")
    def v4l2_formats(index: int) -> dict[str, Any]:
        """``v4l2-ctl -d /dev/video{index} --list-formats-ext`` (parsed)."""
        return v4l2_tools.list_formats_for_index(int(index))

    @router.get("/api/devices/v4l2/controls")
    def v4l2_controls(index: int) -> dict[str, Any]:
        """UVC/V4L2 controls on ``/dev/video{index}`` (brightness, zoom, exposure, …)."""
        out = v4l2_camera_controls.list_camera_controls(int(index))
        if not out.get("ok"):
            raise HTTPException(status_code=503, detail=out.get("error") or "list-ctrls failed")
        return out

    @router.post("/api/devices/v4l2/controls/set")
    def v4l2_controls_set(body: V4l2SetControlsBody) -> dict[str, Any]:
        out = v4l2_camera_controls.set_camera_controls(int(body.index), body.controls)
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "set-ctrl failed")
        return out

    @router.post("/api/devices/v4l2/controls/reset")
    def v4l2_controls_reset(index: int) -> dict[str, Any]:
        out = v4l2_camera_controls.reset_camera_controls(int(index))
        if not out.get("ok"):
            raise HTTPException(status_code=400, detail=out.get("error") or "reset failed")
        return out

    @router.post("/api/devices/probe")
    def probe_source(body: ProbeSourceBody) -> dict[str, Any]:
        """Probe a local ``/dev/videoN`` or network (RTSP/HTTP) source — OK or Error."""
        return _probe_video_source(body.source)

    @router.get("/api/devices/serial")
    def serial_port_candidates() -> dict[str, Any]:
        """ttyUSB* / ttyACM* for mmWave auto-detect."""
        return v4l2_tools.list_serial_port_candidates()

    return router
