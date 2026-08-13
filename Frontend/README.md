# SCAN-U Dashboard

Dashboard de monitoreo en tiempo real para dispositivos Jetson con detección de personas/armas mediante IA.

## 🏗️ Arquitectura

```
[Jetson Edge Devices] 
    ↓ RTSP/WebRTC + heartbeats
[Central Server (GPU+) - FastAPI]
    - Person Detection (YOLOv8/v10)
    - Weapon Detection (YOLO custom)
    - Multi-Object Tracking (ByteTrack/DeepSORT)
    - Event Engine (unsafe detection)
    ↓ WebSocket: ws://localhost:8000/ws/dashboard
[Dashboard Frontend - React + Tailwind]
    - Renderizado de detecciones en tiempo real
    - Visualización de trayectorias
    - Panel de alertas críticas
```

## 🚀 Frontend (Este Proyecto)

### Stack
- **React** + **TypeScript**
- **Tailwind CSS v4**
- **WebSocket** (cliente único con reconexión automática)
- **Vite** (dev server ya corriendo en el entorno)

### Estado Actual

✅ **LISTO PARA BACKEND:**
- WebSocket client configurado (`ws://localhost:8000/ws/dashboard`)
- Protocolo completamente documentado (ver `BACKEND_INTEGRATION.md`)
- Renderizado de bounding boxes, IDs y trails optimizado
- Reconexión automática con exponential backoff
- Mock data eliminado (solo dispositivos offline cuando backend no está)

### Ejecutar

```bash
# Instalar dependencias (si es necesario)
pnpm install

# El dev server ya está corriendo en el entorno Make
# No ejecutar manualmente vite/npm run dev
```

### Archivos Clave

- **`src/app/api/client.ts`** - Cliente WebSocket + protocolo completo
- **`src/app/components/DetectionOverlay.tsx`** - Renderizado de boxes/IDs/trails con SVG
- **`src/app/components/MainCamera.tsx`** - Vista principal de cámaras
- **`BACKEND_INTEGRATION.md`** - **📖 Guía completa para implementar el backend**

## 🤖 Backend (Pendiente - Ver BACKEND_INTEGRATION.md)

### Checklist
- [ ] Endpoint WebSocket: `ws://localhost:8000/ws/dashboard`
- [ ] YOLOv8/v10 para person detection
- [ ] YOLO custom para weapon detection
- [ ] ByteTrack o DeepSORT para tracking persistente
- [ ] Normalización de bboxes a 0-1
- [ ] Acumulación de trails (últimos 20 puntos)
- [ ] Event engine (persona + arma → unsafe)
- [ ] Streaming a ~30 FPS vía WebSocket

### Ejemplo Backend Mínimo

```python
# main.py
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Enviar snapshot inicial
    await websocket.send_json({
        "type": "snapshot",
        "data": {
            "devices": [...],  # ver BACKEND_INTEGRATION.md
            "metrics": {...},
            "status": {...},
            "alerts": [],
            "detections": {}
        }
    })
    
    # 2. Loop: procesar frames y enviar detections
    while True:
        detections = await process_frame()  # YOLO + ByteTrack
        await websocket.send_json({
            "type": "detections",
            "data": {
                "deviceId": "jetson-01",
                "detections": detections
            }
        })
        await asyncio.sleep(1/30)  # ~30 FPS
```

## 📊 Features Implementadas

### UI
- ✅ Sidebar con lista de Jetson devices + indicadores de estado
- ✅ Vista principal de cámara (RGB/Thermal/mmWave)
- ✅ Split view (RGB + Thermal simultáneos)
- ✅ Panel de métricas (FPS, latency, CPU/GPU/Memory)
- ✅ Console bar con alertas en tiempo real
- ✅ Modales de configuración (Device + Admin)

### Visualización
- ✅ Bounding boxes con colores por tipo (persona/arma)
- ✅ Tracking IDs persistentes
- ✅ Trayectorias (trails) con últimos 20 puntos
- ✅ Highlight de personas "unsafe" (con arma)
- ✅ Animación de pulso en detecciones de armas
- ✅ Overlay mmWave con point cloud

### Lógica
- ✅ WebSocket con reconexión automática
- ✅ Fallback a dispositivos offline cuando backend no responde
- ✅ Toggle de visualización (boxes/IDs/trails)
- ✅ Cambio de modo de operación (central/fallback/local)
- ✅ Indicador visual "Waiting for Backend"

## 🔧 Configuración

### WebSocket URL Override

```typescript
// En el navegador o config
window.__SCANU_WS_URL__ = "ws://custom-server:8080/ws/dashboard";
```

### Variables de Entorno

```env
# No hay variables de entorno en frontend
# Todo se configura vía WebSocket desde el backend
```

## 📖 Documentación Completa

**Para implementar el backend:** Lee `BACKEND_INTEGRATION.md`

Incluye:
- Protocolo WebSocket completo (todos los mensajes)
- Ejemplo de pipeline YOLO + ByteTrack
- Checklist de implementación por fases
- Tests de conexión
- Campos críticos (normalización, trails, IDs)

## 🎨 UI Preview

```
┌─────────────────────────────────────────────────────────────┐
│ [SCAN-U] [NORMAL] [Mode: Central/Fallback/Local] [Alerts]  │
├───┬─────────────────────────────────────────────────────────┤
│ J │  [RGB] [Thermal] [mmWave]  [Boxes] [IDs] [Trails]      │
│ e │                                                          │
│ t │  ┌────────────────────────────────────────────┐         │
│ s │  │  🔴 LIVE          [Camera Feed]            │         │
│ o │  │                                            │         │
│ n │  │     ┌─────┐ T-01 92%                      │         │
│ s │  │     │ ░░░ │  (bounding box + trail)       │         │
│   │  │     └─────┘                                │         │
│   │  └────────────────────────────────────────────┘         │
│   │                                                          │
├───┴──────────────────────────────────────────────────────────┤
│ [Console] Weapon detected • Thermal anomaly • Track acquired │
└──────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

```bash
# Test WebSocket desde Python
python test_ws.py  # Ver BACKEND_INTEGRATION.md

# Verificar reconexión
# 1. Arrancar frontend (ya corriendo)
# 2. Arrancar backend
# 3. Detener backend → ver "Waiting for Backend"
# 4. Re-arrancar backend → reconexión automática
```

## 📝 Notas

- **No hay datos mock de tracking:** El frontend solo muestra dispositivos offline hasta que el backend envíe datos reales
- **Normalización 0-1:** Todas las bboxes y trails están normalizadas (backend debe enviarlas así)
- **Trails server-side:** El backend debe calcular y acumular las trayectorias
- **IDs persistentes:** El tracker (ByteTrack/DeepSORT) asigna IDs que persisten entre frames
- **~30 FPS:** El backend debe pushear frames procesados a alta frecuencia

---

**Next Step:** Implementar el backend siguiendo `BACKEND_INTEGRATION.md`
