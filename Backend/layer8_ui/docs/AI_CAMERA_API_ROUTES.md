# Layer 8 — AI Camera API routes (frontend integration)

Server base:

```bash
export BASE=http://127.0.0.1:8088
```

This API namespace maps to the webcam + weapon-inference pipeline.

## Profiles

### Get profiles

`GET /api/ai_camera/profiles`

Returns a **list** of profiles (`id`, `name`, `description`, `values`). For the on-disk dict shape, use `GET /api/model/profiles`.

```bash
curl -sS "$BASE/api/ai_camera/profiles"
```

Response:

```json
{
  "profiles": [
    {
      "id": "profile_1080p_30fps",
      "name": "1080p 30fps",
      "description": "1080p capture at 30 FPS with balanced settings.",
      "values": {
        "webcam_width": 1920,
        "webcam_height": 1080,
        "fps": 30
      }
    }
  ]
}
```

### Apply profile by ID (existing route)

`POST /api/model/profiles/apply`

```bash
curl -sS -X POST "$BASE/api/model/profiles/apply" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "profile_1080p_15fps"
  }'
```

Returns the full updated **`ui_settings`** document (same as `GET /api/ai_camera/config` after apply).

### Apply profile by name (new route)

`POST /api/ai_camera/profiles/apply_by_name`

Name matching is **case-insensitive** and compares to each profile’s display **`name`** (stored as `label` in `model_profiles.json`).

```bash
curl -sS -X POST "$BASE/api/ai_camera/profiles/apply_by_name" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "1080p 15fps"
  }'
```

Success response:

```json
{
  "ok": true,
  "applied_profile_id": "profile_1080p_15fps",
  "applied_profile_name": "1080p 15fps",
  "settings": { "...": "full ui_settings payload" }
}
```

If multiple profiles share the same **name** (after trim + case-fold), returns **409** with `detail.matching_ids`.

## AI camera config

### Get config

`GET /api/ai_camera/config`

Returns the full **`ui_settings`** JSON (thermal, webcam, mmwave, etc.), same as the dashboard persistence layer.

```bash
curl -sS "$BASE/api/ai_camera/config"
```

### Update config (partial merge into `webcam`)

`PUT /api/ai_camera/config`

Only the **`webcam`** key is merged when present; other top-level sections are unchanged.

```bash
curl -sS -X PUT "$BASE/api/ai_camera/config" \
  -H "Content-Type: application/json" \
  -d '{
    "webcam": {
      "webcam_device": 2,
      "webcam_width": 1280,
      "webcam_height": 720,
      "fps": 30,
      "weapon_conf": 0.25,
      "weapon_gun_conf": 0.25,
      "weapon_gun_imgsz": 512
    }
  }'
```

## Runner controls

### Status

`GET /api/ai_camera/status`

```bash
curl -sS "$BASE/api/ai_camera/status"
```

### Run

`POST /api/ai_camera/run`

```bash
curl -sS -X POST "$BASE/api/ai_camera/run"
```

### Stop

`POST /api/ai_camera/stop`

```bash
curl -sS -X POST "$BASE/api/ai_camera/stop"
```

### Restart

`POST /api/ai_camera/restart`

```bash
curl -sS -X POST "$BASE/api/ai_camera/restart"
```

## Live preview stream

`GET /api/ai_camera/preview/live`

When the webcam runner is on, MJPEG matches WebRTC sources **without** reading `webcam.live_frame` from disk (avoids stale-file flashes): **BGR IPC** → re-encode as JPEG, else **JPEG IPC**. Runner off: shared V4L2 `WebcamSharedStream` (same as WebRTC’s non-runner fallback).

```bash
curl -sS --max-time 2 "$BASE/api/ai_camera/preview/live" -o /tmp/ai_camera_preview_mjpeg.bin
```

## WebRTC stream (AI boxes)

WebRTC endpoint aliases (same backend behavior):

- `POST /api/ai_camera/webrtc/offer`
- `POST /api/webrtc/ai_camera/offer`
- `POST /api/webrtc/webcam/offer` (legacy)

Request body:

```json
{
  "sdp": "v=0...",
  "type": "offer"
}
```

Response body:

```json
{
  "sdp": "v=0...",
  "type": "answer"
}
```

Example JS call:

```javascript
const offerResp = await fetch("/api/ai_camera/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
  }),
});
```

## Useful existing routes (still valid)

- `GET /api/model/profiles`
- `POST /api/model/profiles/snapshot`
- `POST /api/model/profiles/sync_from_config`
- `GET /api/dashboard/metrics` (unsafe/gun/person summary JSON)
- `POST /api/run/webcam`, `POST /api/stop/webcam`, `POST /api/restart/webcam`
