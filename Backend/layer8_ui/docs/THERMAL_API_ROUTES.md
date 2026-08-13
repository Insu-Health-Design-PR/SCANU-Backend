# Layer 8 — Thermal API routes (frontend integration)

**Server:** SCANU Layer8 UI (`uvicorn layer8_ui.app:app`)

Set a base URL for all examples:

```bash
export BASE=http://127.0.0.1:8088
```

All JSON responses use UTF-8. Errors may return `{ "detail": "..." }` (FastAPI) or `{ "ok": false, ... }` depending on the route.

---

## Thermal configuration

### `GET /api/thermal/config`

Returns the current **`thermal`** block from `layer8_ui/ui_settings.json` (persisted).

```bash
curl -sS "$BASE/api/thermal/config"
```

**Example response**

```json
{
  "thermal": {
    "frames": 0,
    "fps": 10.0,
    "video": "",
    "live_frame": "layer8_ui/artifacts/live_thermal.jpg",
    "output": "",
    "thermal_device": 1,
    "thermal_auto_detect": 1,
    "thermal_detect_max_index": 6,
    "thermal_detect_retry_s": 12.0,
    "thermal_width": 640,
    "thermal_height": 480,
    "thermal_fps": 30,
    "panel_w": 640,
    "panel_h": 480
  }
}
```

---

### `PUT /api/thermal/config`

Merge updates into **`thermal`** (partial update). Unknown keys are preserved for keys you omit.

**Body:** `{ "thermal": { ... } }`

```bash
curl -sS -X PUT "$BASE/api/thermal/config" \
  -H "Content-Type: application/json" \
  -d '{
    "thermal": {
      "thermal_device": 0,
      "thermal_width": 160,
      "thermal_height": 120,
      "thermal_fps": 9,
      "panel_w": 640,
      "panel_h": 480,
      "live_frame": "layer8_ui/artifacts/live_thermal.jpg",
      "thermal_auto_detect": 1
    }
  }'
```

Returns the full updated **`ui_settings`** document (same as `GET /api/config` after save).

---

### `POST /api/thermal/auto_configure`

Probes V4L2 devices using current `thermal_*` geometry/FPS settings, picks a working index, saves `thermal_device` and sets `thermal_auto_detect` to `1`.

No JSON body required.

```bash
curl -sS -X POST "$BASE/api/thermal/auto_configure"
```

**Success (200)**

```json
{
  "ok": true,
  "thermal": { "...": "merged config after save" },
  "detected_device": 0
}
```

**Failure (404)** — no usable device found

```json
{
  "ok": false,
  "error": "No working thermal V4L2 device found",
  "thermal": { "...": "previous thermal block" }
}
```

---

## Runner control

### `GET /api/thermal/status`

```bash
curl -sS "$BASE/api/thermal/status"
```

**Example response**

```json
{
  "running": true,
  "pid": 12345,
  "log_tail": "...last log bytes...",
  "log_file": "/path/to/software/layer8_ui/logs/thermal.log"
}
```

---

### `POST /api/thermal/run`

Starts the thermal capture subprocess (writes JPEG to `thermal.live_frame` when configured).

No JSON body required.

```bash
curl -sS -X POST "$BASE/api/thermal/run"
```

**Success (200)**

```json
{
  "ok": true,
  "pid": 12345,
  "command": ["python", "...", "--thermal-device", "0", "..."],
  "cwd": "/path/to/software",
  "software_root": "/path/to/software",
  "log_file": "/path/to/software/layer8_ui/logs/thermal.log"
}
```

**Conflict (409)** — already running or spawn failed `{ "ok": false, "error": "..." }`

---

### `POST /api/thermal/stop`

```bash
curl -sS -X POST "$BASE/api/thermal/stop"
```

**Example response**

```json
{
  "ok": true,
  "stopped_pid": 12345
}
```

---

### `POST /api/thermal/restart`

Stop then start thermal with current saved settings.

```bash
curl -sS -X POST "$BASE/api/thermal/restart"
```

Responses match **`run`** / **409** semantics.

---

## Live preview streaming

### `GET /api/thermal/preview/live`

MJPEG over HTTP (`multipart/x-mixed-replace`). Use as `<img src="...">` in a browser or consume as a multipart stream.

```bash
curl -sS --max-time 2 "$BASE/api/thermal/preview/live" -o /tmp/thermal_preview_mjpeg.bin
```

Equivalent paths (same stream): **`GET /api/preview/live/thermal`**, **`GET /api/preview/live_direct/thermal`**.

---

### `WebSocket /ws/thermal`

Binary JPEG frames (use a WebSocket client in the browser; not plain HTTP GET).

---

## Helpers (hardware discovery)

### `GET /api/devices/v4l2`

List V4L2 groups and suggested thermal/webcam indices.

```bash
curl -sS "$BASE/api/devices/v4l2"
```

### `GET /api/devices/v4l2/formats?index=<n>`

List supported formats/resolutions/FPS for `/dev/video<n>`.

```bash
curl -sS "$BASE/api/devices/v4l2/formats?index=0"
```

---

## Legacy aliases (same behavior)

These still work and mirror thermal-specific routes:

| Thermal-first route        | Equivalent sensor route      |
|---------------------------|------------------------------|
| —                         | `GET /api/command/thermal`   |
| `POST /api/thermal/run`   | `POST /api/run/thermal`      |
| `POST /api/thermal/stop`| `POST /api/stop/thermal`    |
| `POST /api/thermal/restart` | `POST /api/restart/thermal` |

```bash
curl -sS "$BASE/api/command/thermal"
```

---

## Full settings (optional)

To read/write **entire** dashboard config (thermal + webcam + mmwave):

- `GET /api/config`
- `PUT /api/config` with body `{ "settings": { ... } }`

```bash
curl -sS "$BASE/api/config"
```

---

*Routes defined in `layer8_ui/dashboard_routes.py`.*
