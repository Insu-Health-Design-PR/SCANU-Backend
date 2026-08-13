# Backend Integration Guide - SCAN-U Dashboard

## 🎯 Overview

Este dashboard se conecta a un backend FastAPI vía WebSocket. El backend debe procesar streams de Jetson devices y enviar datos procesados listos para renderizar.

---

## 🔌 WebSocket Endpoint

```
WS ws://localhost:8000/ws/dashboard
```

**Configuración:**
- Por defecto: `ws://localhost:8000/ws/dashboard`
- Override en runtime: `window.__SCANU_WS_URL__ = "ws://your-server:port/ws/dashboard"`

---

## 📡 Protocolo WebSocket

### **SERVER → CLIENT** (Backend envía al Dashboard)

#### 1. **Snapshot Completo** (Enviar al conectar + cada 2 segundos)
```json
{
  "type": "snapshot",
  "data": {
    "devices": [
      {
        "id": "jetson-01",
        "name": "Jetson 1",
        "location": "Main Entrance",
        "status": "online",
        "cameras": { "webcam": true, "thermal": true },
        "mmwave": true,
        "lastHeartbeat": "2026-06-01T10:30:45.123Z",
        "fps": 28.5,
        "latencyMs": 72,
        "temperature": 52
      }
    ],
    "metrics": {
      "totalDetections": 243,
      "weaponDetections": 2,
      "personDetections": 48,
      "avgConfidence": 0.87,
      "fps": 28.5,
      "latency": 72,
      "cpuUsage": 65.2,
      "gpuUsage": 78.9,
      "memoryUsage": 62.1
    },
    "status": {
      "mode": "central",
      "state": "normal",
      "backendOnline": true,
      "activeAlerts": 1,
      "timestamp": "2026-06-01T10:30:45.123Z"
    },
    "alerts": [
      {
        "id": "alert-abc123",
        "severity": "critical",
        "type": "weapon_detected",
        "message": "Arma detectada en frame",
        "deviceId": "jetson-01",
        "sensor": "webcam",
        "confidence": 0.89,
        "timestamp": "2026-06-01T10:30:44.000Z",
        "acknowledged": false
      }
    ],
    "detections": {
      "jetson-01": [
        {
          "id": "det-frame-1234-obj-0",
          "type": "person",
          "confidence": 0.87,
          "bbox": [0.25, 0.30, 0.12, 0.28],
          "trackingId": "T-01",
          "unsafe": true,
          "trail": [
            { "x": 0.23, "y": 0.40 },
            { "x": 0.24, "y": 0.41 },
            { "x": 0.25, "y": 0.42 }
          ],
          "timestamp": "2026-06-01T10:30:45.123Z"
        },
        {
          "id": "det-frame-1234-obj-1",
          "type": "weapon",
          "confidence": 0.92,
          "bbox": [0.26, 0.32, 0.05, 0.05],
          "trackingId": "W-01",
          "unsafe": true,
          "trail": [],
          "timestamp": "2026-06-01T10:30:45.123Z"
        }
      ],
      "jetson-02": [],
      "jetson-03": []
    }
  }
}
```

#### 2. **Detections Update** (Enviar cada frame procesado, ~30 FPS)
```json
{
  "type": "detections",
  "data": {
    "deviceId": "jetson-01",
    "detections": [
      {
        "id": "det-frame-1235-obj-0",
        "type": "person",
        "confidence": 0.88,
        "bbox": [0.26, 0.31, 0.12, 0.28],
        "trackingId": "T-01",
        "unsafe": true,
        "trail": [
          { "x": 0.24, "y": 0.41 },
          { "x": 0.25, "y": 0.42 },
          { "x": 0.26, "y": 0.43 }
        ],
        "timestamp": "2026-06-01T10:30:45.156Z"
      }
    ]
  }
}
```

#### 3. **Nueva Alerta** (Cuando detectes evento crítico)
```json
{
  "type": "alert",
  "data": {
    "id": "alert-xyz789",
    "severity": "critical",
    "type": "weapon_detected",
    "message": "Arma detectada en Main Entrance",
    "deviceId": "jetson-01",
    "sensor": "webcam",
    "confidence": 0.92,
    "timestamp": "2026-06-01T10:30:45.200Z",
    "acknowledged": false
  }
}
```

#### 4. **Metrics Update** (Cada 2-5 segundos)
```json
{
  "type": "metrics",
  "data": {
    "totalDetections": 245,
    "weaponDetections": 2,
    "personDetections": 49,
    "avgConfidence": 0.86,
    "fps": 29.2,
    "latency": 68,
    "cpuUsage": 64.8,
    "gpuUsage": 79.3,
    "memoryUsage": 62.5
  }
}
```

#### 5. **Devices Update** (Cuando cambie estado de Jetson)
```json
{
  "type": "devices",
  "data": [
    {
      "id": "jetson-01",
      "name": "Jetson 1",
      "location": "Main Entrance",
      "status": "online",
      "cameras": { "webcam": true, "thermal": true },
      "mmwave": true,
      "lastHeartbeat": "2026-06-01T10:30:50.000Z",
      "fps": 28.5,
      "latencyMs": 72,
      "temperature": 53
    }
  ]
}
```

#### 6. **Status Update** (Cuando cambie estado del sistema)
```json
{
  "type": "status",
  "data": {
    "mode": "fallback",
    "state": "recovery",
    "backendOnline": true,
    "activeAlerts": 3,
    "timestamp": "2026-06-01T10:30:46.000Z"
  }
}
```

---

### **CLIENT → SERVER** (Dashboard envía al Backend)

#### 1. **Seleccionar Device**
```json
{
  "type": "select_device",
  "deviceId": "jetson-02"
}
```

#### 2. **Cambiar Modo de Operación**
```json
{
  "type": "set_mode",
  "mode": "fallback"
}
```
Valores: `"central"` | `"fallback"` | `"local"`

#### 3. **Iniciar Sensor**
```json
{
  "type": "run_sensor",
  "deviceId": "jetson-01",
  "sensor": "webcam"
}
```
Valores de sensor: `"webcam"` | `"thermal"` | `"mmwave"`

#### 4. **Detener Sensor**
```json
{
  "type": "stop_sensor",
  "deviceId": "jetson-01",
  "sensor": "thermal"
}
```

#### 5. **Acknowledge Alert**
```json
{
  "type": "ack_alert",
  "alertId": "alert-abc123"
}
```

---

## 🤖 Procesamiento de IA Requerido (Backend)

### Stack Recomendado

```python
# requirements.txt
fastapi
uvicorn[standard]
websockets
ultralytics      # YOLOv8/v10
supervision      # ByteTrack wrapper
opencv-python
numpy
```

### Pipeline de Procesamiento

```python
from ultralytics import YOLO
from supervision import ByteTrack
from collections import defaultdict, deque

# 1. Cargar modelos
person_detector = YOLO('yolov8n.pt')  # O yolov10n.pt
weapon_detector = YOLO('weapon_yolov8.pt')  # Entrenado custom

# 2. Inicializar tracker
tracker = ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=30,
    minimum_matching_threshold=0.7,
    frame_rate=30
)

# 3. Historiales de trails (max 20 puntos por track)
trails = defaultdict(lambda: deque(maxlen=20))

# 4. Loop de procesamiento
async def process_frame(device_id: str, frame: np.ndarray):
    h, w = frame.shape[:2]
    
    # Detección
    person_results = person_detector(frame, conf=0.5)[0]
    weapon_results = weapon_detector(frame, conf=0.7)[0]
    
    # Tracking (asigna IDs persistentes)
    tracked = tracker.update_with_detections(person_results)
    
    detections = []
    unsafe_track_ids = set()
    
    # Detectar unsafe (persona + arma cercana)
    for weapon_box in weapon_results.boxes:
        wx, wy = weapon_box.xyxy[0][:2].cpu().numpy()
        for person_track in tracked:
            px, py = person_track.xyxy[:2]
            if np.linalg.norm([wx - px, wy - py]) < 100:  # threshold en píxeles
                unsafe_track_ids.add(person_track.tracker_id)
    
    # Construir detections
    for track in tracked:
        x1, y1, x2, y2 = track.xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Normalizar coordenadas 0-1
        bbox = [
            float(x1 / w),
            float(y1 / h),
            float((x2 - x1) / w),
            float((y2 - y1) / h)
        ]
        
        # Acumular trail
        trail_id = f"{device_id}-{track.tracker_id}"
        trails[trail_id].append({
            "x": float(cx / w),
            "y": float(cy / h)
        })
        
        is_unsafe = track.tracker_id in unsafe_track_ids
        
        detections.append({
            "id": f"det-{device_id}-{int(time.time() * 1000)}-{track.tracker_id}",
            "type": "person",
            "confidence": float(track.confidence),
            "bbox": bbox,
            "trackingId": f"T-{track.tracker_id:02d}",
            "unsafe": is_unsafe,
            "trail": list(trails[trail_id]),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Agregar weapon detection si es unsafe
        if is_unsafe:
            for weapon_box in weapon_results.boxes:
                wx1, wy1, wx2, wy2 = weapon_box.xyxy[0].cpu().numpy()
                detections.append({
                    "id": f"det-weapon-{device_id}-{int(time.time() * 1000)}",
                    "type": "weapon",
                    "confidence": float(weapon_box.conf[0]),
                    "bbox": [
                        float(wx1 / w),
                        float(wy1 / h),
                        float((wx2 - wx1) / w),
                        float((wy2 - wy1) / h)
                    ],
                    "trackingId": f"W-{track.tracker_id:02d}",
                    "unsafe": True,
                    "trail": [],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
    
    # Enviar por WebSocket
    await websocket.send_json({
        "type": "detections",
        "data": {
            "deviceId": device_id,
            "detections": detections
        }
    })
    
    # Generar alerta si detectaste unsafe
    if unsafe_track_ids:
        await websocket.send_json({
            "type": "alert",
            "data": {
                "id": f"alert-{int(time.time() * 1000)}",
                "severity": "critical",
                "type": "weapon_detected",
                "message": f"Arma detectada en {device_id}",
                "deviceId": device_id,
                "sensor": "webcam",
                "confidence": max(det["confidence"] for det in detections if det["type"] == "weapon"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "acknowledged": False
            }
        })
```

---

## 📋 Checklist de Implementación Backend

### Fase 1: WebSocket básico
- [ ] Endpoint `ws://localhost:8000/ws/dashboard` funcional
- [ ] Enviar `snapshot` completo al conectar
- [ ] Enviar `snapshot` heartbeat cada 2 segundos
- [ ] Recibir y loggear mensajes del cliente

### Fase 2: Detección YOLO
- [ ] Integrar YOLOv8/v10 para person detection
- [ ] Integrar YOLO custom para weapon detection
- [ ] Normalizar bboxes a 0-1
- [ ] Enviar `detections` cada frame (~30 FPS)

### Fase 3: Multi-Object Tracking
- [ ] Implementar ByteTrack o DeepSORT
- [ ] Asignar IDs persistentes (`T-01`, `T-02`, etc.)
- [ ] Acumular trails (últimos 20 centros)
- [ ] Mantener trails en memoria

### Fase 4: Event Engine
- [ ] Detectar persona + arma cercana → `unsafe: true`
- [ ] Generar alertas críticas automáticas
- [ ] Enviar mensaje tipo `"alert"`

### Fase 5: Métricas y Estado
- [ ] Calcular métricas agregadas (FPS, latency, detecciones totales)
- [ ] Monitorear estado de Jetson devices (heartbeats)
- [ ] Actualizar `status.state` (normal/fallback/recovery/fault)
- [ ] Enviar `metrics` y `devices` periódicamente

---

## 🧪 Testing

### Test básico de conexión:
```python
# test_ws.py
import asyncio
import websockets
import json

async def test_backend():
    uri = "ws://localhost:8000/ws/dashboard"
    async with websockets.connect(uri) as ws:
        # Esperar snapshot inicial
        msg = await ws.recv()
        data = json.loads(msg)
        print(f"Received: {data['type']}")
        assert data['type'] == 'snapshot'
        
        # Seleccionar device
        await ws.send(json.dumps({
            "type": "select_device",
            "deviceId": "jetson-01"
        }))
        
        # Esperar detections
        for i in range(10):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Frame {i}: {len(data.get('data', {}).get('detections', []))} detections")

asyncio.run(test_backend())
```

---

## 🚨 Campos Críticos

### ⚠️ Bounding Boxes (NORMALIZADAS 0-1)
```python
# ❌ INCORRECTO (coordenadas absolutas)
bbox = [150, 200, 300, 400]

# ✅ CORRECTO (normalizadas)
bbox = [
    x1 / frame_width,      # x
    y1 / frame_height,     # y
    (x2 - x1) / frame_width,   # width
    (y2 - y1) / frame_height   # height
]
```

### ⚠️ Trails (últimos 20 centros normalizados)
```python
# Acumular CENTRO del bbox, no esquina
center_x = (x1 + x2) / 2 / frame_width
center_y = (y1 + y2) / 2 / frame_height

trails[track_id].append({"x": center_x, "y": center_y})
trails[track_id] = trails[track_id][-20:]  # Mantener solo últimos 20
```

### ⚠️ Tracking IDs (persistentes entre frames)
```python
# ❌ INCORRECTO (ID nuevo cada frame)
trackingId = f"T-{random.randint(1, 99)}"

# ✅ CORRECTO (ID persistente del tracker)
trackingId = f"T-{tracker_object.id:02d}"
```

---

## 📞 Soporte

- **Frontend listo**: Cliente WebSocket en `/src/app/api/client.ts`
- **Protocolo documentado**: Este archivo + comentarios en `client.ts:1-44`
- **Renderizado**: `DetectionOverlay.tsx` ya dibuja boxes/IDs/trails correctamente

**El frontend NO genera datos fake de tracking.** Todo debe venir del backend.

Cuando el WebSocket esté offline, el dashboard muestra dispositivos offline sin detecciones.
