"""API-side Global ID / Re-ID association service.

Polls Front + Back metrics JSON (and optional IPC frames for crops), runs
``GlobalIDManager``, and writes ``global_person_ids.json`` for UI / overlay.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from weapon_ai.reid.config import ReIDConfig
from weapon_ai.reid.embeddings import MockReIDEmbedder, PersonReIDEmbedder, TrackEmbeddingCache
from weapon_ai.reid.global_manager import GlobalIDManager, LocalObservation, bbox_lateral_norm

logger = logging.getLogger(__name__)


def scale_xyxy(
    bbox: tuple[int, int, int, int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    """Map a box from overlay/metrics space onto a (possibly 4K) crop frame."""
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return bbox
    if src_w == dst_w and src_h == dst_h:
        return bbox
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


class GlobalIDService:
    """Background associator living in the FastAPI process."""

    def __init__(
        self,
        layer8_dir: Path,
        *,
        config: ReIDConfig | None = None,
        state_path: Path | None = None,
        front_metrics: Path | None = None,
        back_metrics: Path | None = None,
        use_mock_embedder: bool = False,
        poll_interval_s: float = 0.2,
    ) -> None:
        self.layer8_dir = Path(layer8_dir).resolve()
        self.config = config or ReIDConfig()
        self.state_path = Path(
            state_path
            or (self.layer8_dir / "configs" / "global_person_ids.json")
        )
        self.front_metrics = Path(
            front_metrics
            or (self.layer8_dir / "configs" / "live_threat_metrics.json")
        )
        # multi_camera default metrics name used by profiles
        alt_front = self.layer8_dir / "configs" / "live_multi_camera_threat_metrics.json"
        self.back_metrics = Path(
            back_metrics
            or (self.layer8_dir / "configs" / "live_multi_camera_threat_metrics.json")
        )
        # Front webcam often writes live_multi_camera_threat_metrics when using multi-cam profile;
        # also try live_threat_metrics. Service accepts explicit paths from settings.
        self._alt_front = self.layer8_dir / "configs" / "live_threat_metrics.json"
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.manager = GlobalIDManager(self.config)
        if use_mock_embedder:
            embedder: PersonReIDEmbedder | MockReIDEmbedder = MockReIDEmbedder()
        else:
            try:
                embedder = PersonReIDEmbedder(
                    device=str(self.config.embed_device),
                    backend=str(getattr(self.config, "embed_backend", "auto") or "auto"),
                    model_name=str(getattr(self.config, "embed_model", "osnet_x0_25") or "osnet_x0_25"),
                    weights_path=str(getattr(self.config, "embed_weights", "") or ""),
                    onnx_path=str(getattr(self.config, "embed_onnx", "") or ""),
                    engine_path=str(getattr(self.config, "embed_engine", "") or ""),
                    input_h=int(getattr(self.config, "embed_input_h", 256) or 256),
                    input_w=int(getattr(self.config, "embed_input_w", 128) or 128),
                )
                logger.info(
                    "GlobalIDService embedder backend=%s device=%s dim=%s",
                    embedder.backend,
                    embedder.device,
                    getattr(embedder, "feature_dim", "?"),
                )
            except Exception as exc:
                logger.warning("GlobalIDService: embedder init failed (%s); using mock", exc)
                embedder = MockReIDEmbedder()
        self.cache = TrackEmbeddingCache(
            embedder,
            interval_s=float(self.config.embed_interval_s),
            max_history=int(self.config.max_embedding_history),
            min_box_px=int(self.config.embed_min_box_px),
        )
        self._frame_providers: dict[str, Any] = {}
        self._logged_full_res: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_snapshot: dict[str, Any] = {"persons": [], "cameras": {}}
        self._enabled = bool(self.config.enable)

    def set_frame_provider(self, camera_id: str, provider: Any) -> None:
        """Register ``provider.latest_bgr() -> np.ndarray | None`` for crops."""
        self._frame_providers[str(camera_id)] = provider

    def configure(self, config: ReIDConfig) -> None:
        with self._lock:
            self.config = config
            self.manager.config = config
            self._enabled = bool(config.enable)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="global-id-service", daemon=True)
        self._thread.start()
        logger.info(
            "GlobalIDService started (enable=%s, state=%s)",
            self._enabled,
            self.state_path,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_snapshot)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._enabled:
                    snap = self.tick()
                    with self._lock:
                        self._last_snapshot = snap
                    _atomic_write_json(self.state_path, snap)
            except Exception as exc:
                logger.warning("GlobalIDService tick failed: %s", exc)
            self._stop.wait(self.poll_interval_s)

    def tick(self, *, now: float | None = None) -> dict[str, Any]:
        """One association cycle (also used by tests)."""
        ts = float(time.time() if now is None else now)
        cam_front = str(self.config.camera_front)
        cam_back = str(self.config.camera_back)
        observations: list[LocalObservation] = []
        observations.extend(self._observations_from_metrics(self.front_metrics, cam_front, ts))
        if not observations:
            # Fallback: some profiles write front metrics to the multi-cam filename only when
            # a single camera runs — still try classic live_threat_metrics for front.
            observations.extend(self._observations_from_metrics(self._alt_front, cam_front, ts))
        observations.extend(self._observations_from_metrics(self.back_metrics, cam_back, ts))

        # Prefer webcam metrics path from settings if front file empty: handled by callers via paths.

        with self._lock:
            snap = self.manager.update_observations(observations, now=ts)
            self._last_snapshot = snap
            return snap

    def _observations_from_metrics(
        self, metrics_path: Path, camera_id: str, ts: float
    ) -> list[LocalObservation]:
        data = _read_json(metrics_path)
        if not data:
            return []
        tracks = data.get("byte_tracks") or []
        if not isinstance(tracks, list):
            return []
        frame, used_full = self._latest_frame(camera_id)
        src_w = _positive_int(data.get("frame_w"))
        src_h = _positive_int(data.get("frame_h"))
        out: list[LocalObservation] = []
        active: set[int] = set()
        for row in tracks:
            if not isinstance(row, dict):
                continue
            try:
                local_id = int(row.get("display_id") or row.get("track_id") or 0)
            except (TypeError, ValueError):
                continue
            if local_id <= 0:
                continue
            active.add(local_id)
            bbox = self._parse_bbox(row)
            if bbox is not None and frame is not None and used_full:
                fh, fw = int(frame.shape[0]), int(frame.shape[1])
                if src_w and src_h:
                    bbox = scale_xyxy(bbox, src_w, src_h, fw, fh)
                elif fw >= 2560 and fh >= 1440:
                    # Older metrics without frame_w/h: overlay was 1080p, crop is 4K.
                    bbox = scale_xyxy(bbox, 1920, 1080, fw, fh)
            depth = row.get("depth_m")
            try:
                depth_m = float(depth) if depth is not None else None
            except (TypeError, ValueError):
                depth_m = None
            lateral_norm = None
            raw_lat = row.get("lateral_norm")
            try:
                lateral_norm = float(raw_lat) if raw_lat is not None else None
            except (TypeError, ValueError):
                lateral_norm = None
            if lateral_norm is None and bbox is not None and src_w > 0:
                lateral_norm = bbox_lateral_norm(bbox, src_w)
            visual = str(row.get("visual_state") or "")
            wconf = float(row.get("weapon_gun_conf") or 0.0)
            weapon = visual in {"armed_gun", "armed_concealed"} or wconf > 0.0
            emb = None
            if frame is not None and bbox is not None:
                emb = self.cache.maybe_update(camera_id, local_id, frame, bbox, now=ts)
            if emb is None:
                emb = self.cache.representative(camera_id, local_id)
            # Metrics may already carry an embedding (infer-side publish)
            raw_emb = row.get("reid_embedding")
            if emb is None and isinstance(raw_emb, list) and raw_emb:
                try:
                    emb = np.asarray(raw_emb, dtype=np.float32)
                except Exception:
                    emb = None
            out.append(
                LocalObservation(
                    camera_id=camera_id,
                    local_track_id=local_id,
                    embedding=emb,
                    bbox=bbox,
                    lateral_norm=lateral_norm,
                    depth_m=depth_m,
                    weapon_detected=bool(weapon),
                    weapon_confidence=max(wconf, 0.5 if weapon and wconf <= 0 else wconf),
                    timestamp=float(data.get("ts") or ts),
                )
            )
        self.cache.prune_missing(camera_id, active)
        return out

    def _latest_frame(self, camera_id: str) -> tuple[np.ndarray | None, bool]:
        """Return ``(frame, used_full_res)``. Prefers the 4K sidecar when present."""
        provider = self._frame_providers.get(camera_id)
        if provider is None:
            return None, False
        try:
            if hasattr(provider, "latest_full_bgr"):
                full = provider.latest_full_bgr()
                if full is not None and getattr(full, "size", 0):
                    if camera_id not in self._logged_full_res:
                        logger.info(
                            "GlobalIDService: 4K Re-ID crops for %s (%sx%s)",
                            camera_id,
                            int(full.shape[1]),
                            int(full.shape[0]),
                        )
                        self._logged_full_res.add(camera_id)
                    return full, True
            if callable(provider):
                preview = provider()
                return preview, False
            if hasattr(provider, "latest_bgr"):
                return provider.latest_bgr(), False
            if hasattr(provider, "get_latest_bgr"):
                return provider.get_latest_bgr(), False
        except Exception:
            return None, False
        return None, False

    @staticmethod
    def _parse_bbox(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
        bbox = row.get("bbox") or row.get("xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            except (TypeError, ValueError):
                return None
        keys = ("x1", "y1", "x2", "y2")
        if all(k in row for k in keys):
            try:
                return int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            except (TypeError, ValueError):
                return None
        return None


def _positive_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


# Process-wide singleton for API routes
_SERVICE: GlobalIDService | None = None
_SERVICE_LOCK = threading.Lock()


def get_global_id_service(layer8_dir: Path | None = None) -> GlobalIDService | None:
    return _SERVICE


def _resolve_metrics_path(layer8_dir: Path, raw: str | None, fallback: str) -> Path:
    """Resolve metrics JSON under repo root or layer8_ui."""
    layer8_dir = Path(layer8_dir).resolve()
    root = layer8_dir.parent if layer8_dir.name == "layer8_ui" else layer8_dir
    val = str(raw or fallback).strip() or fallback
    p = Path(val).expanduser()
    if p.is_absolute():
        return p
    if val.startswith("layer8_ui/"):
        return (root / val).resolve()
    cand = (layer8_dir / val).resolve()
    if cand.exists() or val.startswith("configs/"):
        return cand
    return (root / val).resolve()


def init_global_id_service(
    layer8_dir: Path,
    settings: dict[str, Any] | None = None,
    *,
    start: bool = True,
) -> GlobalIDService:
    global _SERVICE
    cfg = ReIDConfig.from_settings(settings)
    layer8_dir = Path(layer8_dir).resolve()
    graw = (settings or {}).get("global_id") if isinstance(settings, dict) else {}
    graw = graw if isinstance(graw, dict) else {}
    state_rel = str(graw.get("state_json") or "layer8_ui/configs/global_person_ids.json")
    state_path = _resolve_metrics_path(layer8_dir, state_rel, "configs/global_person_ids.json")

    front_m = _resolve_metrics_path(
        layer8_dir,
        str((settings or {}).get("webcam", {}).get("metrics_json") or "")
        if isinstance(settings, dict)
        else "",
        "configs/live_threat_metrics.json",
    )
    back_m = _resolve_metrics_path(
        layer8_dir,
        str((settings or {}).get("multi_camera", {}).get("metrics_json") or "")
        if isinstance(settings, dict)
        else "",
        "configs/live_multi_camera_threat_metrics.json",
    )

    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = GlobalIDService(
                layer8_dir,
                config=cfg,
                state_path=state_path,
                front_metrics=front_m,
                back_metrics=back_m,
            )
        else:
            _SERVICE.configure(cfg)
            _SERVICE.front_metrics = front_m
            _SERVICE.back_metrics = back_m
            _SERVICE.state_path = state_path
        if start:
            _SERVICE.start()
        return _SERVICE
