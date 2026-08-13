# Layer 8 — mmWave API routes (frontend integration)

**Server:** SCANU Layer8 UI (`uvicorn layer8_ui.app:app`)

Set a base URL for all examples:

```bash
export BASE=http://127.0.0.1:8088
```

All JSON responses use UTF-8. Errors may return `{ "detail": "..." }` (FastAPI) or `{ "ok": false, ... }` depending on the route.

---

## mmWave configuration

### `GET /api/mmwave/config`

Returns the current **`mmwave`** block from `layer8_ui/ui_settings.json` (persisted).

```bash
curl -sS "$BASE/api/mmwave/config"
```

**Example response**

```json
{
  "mmwave": {
    "config": "layer1_radar/examples/configs/stable_tracking_indoor2_low_uart.cfg",
    "cli_port": "/dev/ttyUSB0",
    "data_port": "/dev/ttyUSB1",
    "live_frame": "layer8_ui/artifacts/live_mmwave.jpg",
    "output": "layer8_ui/artifacts/mmwave_frames.json"
  }
}
```

---

### `PUT /api/mmwave/config`

Merge updates into **`mmwave`** (partial update). Keys you omit are left unchanged.

**Body:** `{ "mmwave": { ... } }`

```bash
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

Returns the full updated **`ui_settings`** document (same as `GET /api/config` after save).

---

### `POST /api/mmwave/auto_configure`

Guesses **`cli_port`** and **`data_port`** from `/dev/ttyUSB*` / `/dev/ttyACM*` (same logic as `GET /api/devices/serial` and the dashboard **Auto-detect** button).

No JSON body required.

```bash
curl -sS -X POST "$BASE/api/mmwave/auto_configure"
```

**Success (200)**

```json
{
  "ok": true,
  "mmwave": { "...": "merged config after save" },
  "suggested_cli": "/dev/ttyUSB0",
  "suggested_data": "/dev/ttyUSB1",
  "ports": ["/dev/ttyUSB0", "/dev/ttyUSB1"]
}
```

**Failure (404)** — serial discovery reported failure (unusual; `list_serial_port_candidates` normally succeeds with defaults).

---

## Runner control

### `GET /api/mmwave/status`

```bash
curl -sS "$BASE/api/mmwave/status"
```

**Example response**

```json
{
  "running": false,
  "pid": null,
  "log_tail": "",
  "log_file": "/path/to/software/layer8_ui/logs/mmwave.log"
}
```

---

### `POST /api/mmwave/run`

Starts the mmWave capture subprocess (writes JPEG to `mmwave.live_frame` and JSON to `mmwave.output` when configured).

```bash
curl -sS -X POST "$BASE/api/mmwave/run"
```

**Conflict (409)** — already running or spawn failed: `{ "ok": false, "error": "..." }`

---

### `POST /api/mmwave/stop`

```bash
curl -sS -X POST "$BASE/api/mmwave/stop"
```

---

### `POST /api/mmwave/restart`

Stop then start with current saved settings.

```bash
curl -sS -X POST "$BASE/api/mmwave/restart"
```

---

## Live preview and output

### `GET /api/mmwave/preview/live`

MJPEG over HTTP from **`mmwave.live_frame`**.

```bash
curl -sS --max-time 2 "$BASE/api/mmwave/preview/live" -o /tmp/mmwave_preview_mjpeg.bin
```

Equivalent paths (same stream): **`GET /api/preview/live/mmwave`**, **`GET /api/preview/live_direct/mmwave`**.

---

### `GET /api/mmwave/preview/output`

Serves the JSON file configured under **`mmwave.output`** when it exists.

Equivalent: **`GET /api/preview/output/mmwave`**

```bash
curl -sS "$BASE/api/mmwave/preview/output"
```

---

## Helpers (hardware discovery)

### `GET /api/devices/serial`

List serial candidates and suggested CLI/data ports.

```bash
curl -sS "$BASE/api/devices/serial"
```

---

## Legacy aliases (same behavior)

| mmWave-first route | Equivalent sensor route |
|--------------------|-------------------------|
| `POST /api/mmwave/run` | `POST /api/run/mmwave` |
| `POST /api/mmwave/stop` | `POST /api/stop/mmwave` |
| `POST /api/mmwave/restart` | `POST /api/restart/mmwave` |
| `GET /api/mmwave/status` | `GET /api/status/mmwave` |

```bash
curl -sS "$BASE/api/command/mmwave"
```

---

## Full settings (optional)

To read/write **entire** dashboard config (thermal + webcam + mmwave):

- `GET /api/config`
- `PUT /api/config` with body `{ "settings": { ... } }`

---

*Routes defined in `layer8_ui/dashboard_routes.py`.*
