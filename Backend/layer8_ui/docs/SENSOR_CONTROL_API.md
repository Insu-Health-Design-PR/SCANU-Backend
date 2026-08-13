# Layer 8 — Sensor Control API (Thermal + AI Camera + mmWave)

Single integration sheet for frontend control flows. Routes are implemented in `layer8_ui/dashboard_routes.py`.

```bash
export BASE=http://127.0.0.1:8088
```

---

## 1) Global status and generic control

### Get all statuses

`GET /api/status`

```bash
curl -sS "$BASE/api/status"
```

### Get one sensor status

`GET /api/status/{sensor}` where `sensor` is `thermal` | `webcam` | `mmwave`

```bash
curl -sS "$BASE/api/status/thermal"
curl -sS "$BASE/api/status/webcam"
curl -sS "$BASE/api/status/mmwave"
```

### Status stream (SSE)

`GET /api/status/stream` — Server-Sent Events; each event is JSON for all three sensors (polls ~1 Hz). Use this from `EventSource` in the browser. There is **no** `/ws/events` WebSocket today.

```bash
curl -sS -N "$BASE/api/status/stream"
```

### Generic run / stop / restart

```bash
curl -sS -X POST "$BASE/api/run/thermal"
curl -sS -X POST "$BASE/api/stop/thermal"
curl -sS -X POST "$BASE/api/restart/thermal"
```

Same pattern for `webcam` and `mmwave`.

### Run / stop / restart all

```bash
curl -sS -X POST "$BASE/api/run_all"
curl -sS -X POST "$BASE/api/stop_all"
curl -sS -X POST "$BASE/api/restart_all"
```

---

## 2) Thermal-first API

### Config

```bash
curl -sS "$BASE/api/thermal/config"

curl -sS -X PUT "$BASE/api/thermal/config" \
  -H "Content-Type: application/json" \
  -d '{
    "thermal": {
      "thermal_device": 0,
      "thermal_width": 640,
      "thermal_height": 480,
      "thermal_fps": 30,
      "fps": 10
    }
  }'
```

`PUT` returns the full updated `ui_settings` document (same as `GET /api/config` after save).

### Auto-configure thermal device

```bash
curl -sS -X POST "$BASE/api/thermal/auto_configure"
```

### Thermal run / stop / restart / status

```bash
curl -sS "$BASE/api/thermal/status"
curl -sS -X POST "$BASE/api/thermal/run"
curl -sS -X POST "$BASE/api/thermal/stop"
curl -sS -X POST "$BASE/api/thermal/restart"
```

### Thermal live preview

- **MJPEG:** `GET /api/thermal/preview/live` (same stream as `GET /api/preview/live/thermal` and legacy `GET /api/preview/live_direct/thermal`)
- **WebSocket** (binary JPEG frames): connect to `WS /ws/thermal` (not HTTP GET)

```bash
curl -sS --max-time 2 "$BASE/api/thermal/preview/live" -o /tmp/thermal_preview_mjpeg.bin
```

More detail: [THERMAL_API_ROUTES.md](./THERMAL_API_ROUTES.md)

---

## 3) AI Camera (webcam + inference) API

### Profiles

```bash
curl -sS "$BASE/api/ai_camera/profiles"

curl -sS -X POST "$BASE/api/ai_camera/profiles/apply_by_name" \
  -H "Content-Type: application/json" \
  -d '{"name":"1080p 15fps"}'
```

Profile by ID (legacy model route, still valid):

```bash
curl -sS -X POST "$BASE/api/model/profiles/apply" \
  -H "Content-Type: application/json" \
  -d '{"id":"profile_1080p_15fps"}'
```

### AI camera config

```bash
curl -sS "$BASE/api/ai_camera/config"

curl -sS -X PUT "$BASE/api/ai_camera/config" \
  -H "Content-Type: application/json" \
  -d '{
    "webcam": {
      "webcam_device": 0,
      "webcam_width": 1920,
      "webcam_height": 1080,
      "fps": 30,
      "weapon_conf": 0.25,
      "weapon_gun_conf": 0.25,
      "weapon_gun_imgsz": 640
    }
  }'
```

### AI camera run / stop / restart / status

```bash
curl -sS "$BASE/api/ai_camera/status"
curl -sS -X POST "$BASE/api/ai_camera/run"
curl -sS -X POST "$BASE/api/ai_camera/stop"
curl -sS -X POST "$BASE/api/ai_camera/restart"
```

(Equivalent to `POST /api/run/webcam`, `stop/webcam`, `restart/webcam`.)

### AI camera live preview

- **MJPEG:** `GET /api/ai_camera/preview/live` (same pipeline as `GET /api/preview/live/webcam` for the AI webcam)
- **WebSocket** JPEG: `WS /ws/webcam`
- **WebRTC** (dashboard `<video>` path):  
  - `POST /api/ai_camera/webrtc/offer`  
  - aliases: `POST /api/webrtc/ai_camera/offer`, `POST /api/webrtc/webcam/offer`

```bash
curl -sS --max-time 2 "$BASE/api/ai_camera/preview/live" -o /tmp/ai_camera_preview_mjpeg.bin
```

WebRTC offer body:

```json
{
  "sdp": "v=0...",
  "type": "offer"
}
```

More detail: [AI_CAMERA_API_ROUTES.md](./AI_CAMERA_API_ROUTES.md)

---

## 4) mmWave-first API

### Config

```bash
curl -sS "$BASE/api/mmwave/config"

curl -sS -X PUT "$BASE/api/mmwave/config" \
  -H "Content-Type: application/json" \
  -d '{
    "mmwave": {
      "cli_port": "/dev/ttyUSB0",
      "data_port": "/dev/ttyUSB1",
      "output": "layer8_ui/artifacts/mmwave_frames.json"
    }
  }'
```

`PUT` returns the full updated `ui_settings` document (same as `GET /api/config` after save). You can still use **`GET/PUT /api/config`** for whole-document edits.

### Auto-detect serial ports (CLI / data)

```bash
curl -sS -X POST "$BASE/api/mmwave/auto_configure"
```

Same heuristics as **`GET /api/devices/serial`** and the dashboard mmWave **Auto-detect** button.

### mmWave run / stop / restart / status

```bash
curl -sS "$BASE/api/mmwave/status"
curl -sS -X POST "$BASE/api/mmwave/run"
curl -sS -X POST "$BASE/api/mmwave/stop"
curl -sS -X POST "$BASE/api/mmwave/restart"
```

(Equivalent to `POST /api/run/mmwave`, `stop/mmwave`, `restart/mmwave`, and `GET /api/status/mmwave`.)

### mmWave live preview and JSON output

- **MJPEG:** `GET /api/mmwave/preview/live` (same stream as `GET /api/preview/live/mmwave` and legacy `GET /api/preview/live_direct/mmwave`)
- **JSON:** `GET /api/mmwave/preview/output` (same file as `GET /api/preview/output/mmwave`)

```bash
curl -sS --max-time 2 "$BASE/api/mmwave/preview/live" -o /tmp/mmwave_preview_mjpeg.bin
curl -sS "$BASE/api/mmwave/preview/output"
```

More detail: [MMWAVE_API_ROUTES.md](./MMWAVE_API_ROUTES.md)

---

## 5) Hardware discovery helpers

```bash
curl -sS "$BASE/api/devices/v4l2"
curl -sS "$BASE/api/devices/v4l2/formats?index=0"
curl -sS "$BASE/api/devices/serial"
```

---

## 6) Other useful endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/command/{sensor}` | Resolved CLI + cwd for thermal / webcam / mmwave |
| `GET /api/dashboard/metrics` | Threat summary JSON (webcam pipeline metrics file) |
| `GET /api/model/profiles` | Model profiles dict (on-disk shape) |
| `PUT /api/model/profiles` | Replace profiles document |
| `GET /api/layer8/module_map` | Which Python modules own each tab |

---

*Run the UI from `software/`: `python3 -m uvicorn layer8_ui.app:app --host 0.0.0.0 --port 8088`*
