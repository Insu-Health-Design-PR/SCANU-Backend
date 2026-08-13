You are a senior product designer, frontend architect, and React/TypeScript engineer.

Create a futuristic, minimal, production-ready UI for a hardware security system called SCAN-U.

PROJECT CONTEXT
SCAN-U is a multi-sensor AI system designed to help detect potential armed intrusions or active-shooter threats before escalation. The system uses multiple sensors and cameras connected to Jetson edge devices, a central GPU/CPU server, AI inference models, mmWave radar point-cloud processing, and a real-time dashboard.

The UI is not a generic camera app. It is a mission-control dashboard for a physical security hardware system that combines:

1. Regular visible camera feeds.
2. Thermal camera feeds.
3. mmWave radar point cloud visualization.
4. AI detection overlays.
5. Jetson device health/status.
6. Server health/status.
7. Alerts, confidence scores, and event logs.
8. Sensor fusion status.
9. Central inference mode and Jetson fallback mode.

The interface should feel like a modern command center: minimal, dark, sharp, clean, high contrast, professional, futuristic, and investor-demo ready. Think: CCTV control center + Tesla UI + Ring camera grid + defense-grade sensor dashboard, but simple and elegant.

SYSTEM ARCHITECTURE CONTEXT

Use the following architecture layers as product context:

LAYER 1 — EDGE LAYER / JETSONS
There are multiple Jetson devices connected to cameras and sensors.

Each Jetson may have:
- Camera Stream 1
- Camera Stream 2
- Thermal camera
- Regular RGB camera
- mmWave radar sensor
- Local fallback inference capability
- Heartbeat connection to the central server

The Jetsons stream data to the central server using RTSP or WebRTC. They also send heartbeat/status updates.

LAYER 2 — CENTRAL SERVER
The central server runs:
- FastAPI Gateway
- Stream Manager
- Preprocessing pipeline
- Resize / Decode / Buffer
- Person YOLO inference
- ROI crop engine
- Weapon YOLO inference
- Multi-object tracker using ByteTrack or DeepSORT
- Event engine for alerting and logging

The central server receives streams from Jetsons, runs inference, manages devices, and sends processed results to the dashboard using WebSocket.

LAYER 3 — CONTROL PLANE
The control plane includes:
- Health Monitor
- Device Registry
- Mode Controller
- Central / Fallback mode manager
- Heartbeat monitoring
- Failure detection
- Auto recovery

The system can operate in:
- NORMAL MODE: central server handles inference.
- FALLBACK MODE: if the server fails, Jetson local inference activates.
- RECOVERY MODE: system restores central mode after server health returns.
- FAULT MODE: device, stream, or sensor failure detected.

LAYER 4 — DASHBOARD LAYER
The dashboard is a React frontend that receives:
- WebSocket video/inference events
- Metrics: FPS, latency, alerts
- Device health
- Jetson status
- Stream status
- mmWave point cloud data
- AI detection overlays
- Tracking IDs and path trails
- Alert logs

MAIN UI GOAL

Design and implement a React + TypeScript frontend dashboard for SCAN-U.

The UI should prioritize 3 main visual panels:

1. Regular Camera View
   - Live RGB/normal camera feed.
   - Detection overlay bounding boxes.
   - Person IDs.
   - Weapon detection highlight.
   - Unsafe person indicator.
   - Stream status badge.
   - FPS and latency mini metrics.

2. Thermal Camera View
   - Live thermal feed.
   - Thermal detection overlay.
   - Human heat signature indicators.
   - Thermal confidence indicator.
   - Stream status badge.
   - FPS and latency mini metrics.

3. mmWave Point Cloud View
   - 2D or 3D point cloud visualization.
   - Moving dots representing detected objects.
   - Movement vectors.
   - Object clusters.
   - Optional trajectory trails.
   - Distance/range rings.
   - Sensor fusion confidence.
   - Presence detection indicator.

Below or beside these 3 main panels, include a dedicated secondary area for:
- Sensor presence status.
- mmWave activity graph.
- AI anomaly/confidence score.
- System console log.
- Jetson status.
- Server status.
- Alert timeline.
- Device controls.

Do not create a cluttered 4-screen layout. The 3 primary panels must dominate the UI:
- Regular camera
- Thermal camera
- mmWave point cloud

The rest should be compact, collapsible, or arranged as smart side/bottom panels.

DESIGN STYLE

Use:
- Dark futuristic theme.
- Minimal layout.
- Professional security dashboard aesthetic.
- Clean spacing.
- Rounded cards.
- Thin borders.
- Subtle glow only for active alerts.
- No childish colors.
- No excessive gradients.
- Use visual hierarchy.

Suggested palette:
- Background: near black / deep navy / graphite.
- Cards: dark charcoal.
- Borders: soft blue-gray.
- Primary accent: cyan / electric blue.
- Warning: amber.
- Critical alert: red.
- Safe/online: green.
- Fallback/recovery: purple or blue.

The UI must feel serious, reliable, and high-tech.

PRIMARY SCREENS / COMPONENTS

Create the following components:

1. Main Dashboard Layout
   - Header with SCAN-U logo/title.
   - Current system mode: NORMAL / FALLBACK / RECOVERY / FAULT.
   - Global threat level indicator.
   - Current time.
   - Connection status.

2. Device Selector / CCTV Grid
   - List of registered Jetson devices.
   - Each device card should show:
     - Device name
     - Location
     - Online/offline status
     - Camera count
     - Thermal status
     - mmWave status
     - Last heartbeat
     - Active alerts
   - UI should feel like Ring/CCTV device listing.

3. Main Sensor View
   Layout with 3 large panels:
   - RGB Camera Panel
   - Thermal Camera Panel
   - mmWave Point Cloud Panel

4. RGB Camera Panel
   - Simulated live video placeholder.
   - Bounding boxes for people.
   - Red bounding box or red glow for weapon detection.
   - Tracking ID labels.
   - Confidence score.
   - Overlay toggle.
   - FPS / latency footer.

5. Thermal Camera Panel
   - Simulated thermal visualization.
   - Heat blobs or gradient overlay.
   - Human presence marker.
   - Thermal confidence.
   - FPS / latency footer.

6. mmWave Point Cloud Panel
   - Canvas/SVG-based point cloud simulation.
   - Dots, clusters, range rings.
   - Movement vectors.
   - Person trajectory line.
   - mmWave object IDs.
   - Basic overlay points for MVP.
   - Enhanced movement vectors for later sprint.
   - Sensor fusion mapping between mmWave object and camera person ID.

7. Alert Center
   - Active alert card.
   - Alert severity.
   - Timestamp.
   - Device source.
   - Sensor source: RGB / thermal / mmWave / fusion.
   - Detection type: person, weapon, anomaly, movement, sensor fault.
   - Confidence score.
   - Acknowledge button.
   - Escalate button.
   - Clear button.

8. Metrics Panel
   - FPS per stream.
   - Latency per stream.
   - AI inference time.
   - WebSocket connection status.
   - GPU usage.
   - CPU usage.
   - RAM usage.
   - Jetson temperature.
   - Server status.

9. Health Monitor Panel
   - Jetson 1 heartbeat.
   - Jetson 2 heartbeat.
   - Stream health.
   - mmWave sensor health.
   - Thermal camera health.
   - RGB camera health.
   - Server health.
   - Mode controller state.

10. Control Panel
   - Start / Stop stream.
   - Restart sensor.
   - Toggle AI overlay.
   - Toggle tracking paths.
   - Toggle mmWave overlay.
   - Switch mode: Central / Fallback.
   - Apply config.
   - Emergency lock / demo mode toggle.

11. Configuration Panel
   - FPS target.
   - Resolution.
   - Confidence threshold.
   - Alert threshold.
   - mmWave sensitivity.
   - Thermal threshold.
   - Model profile selector.
   - Camera selector.

12. System Console Log
   - Real-time logs.
   - Filter by INFO / WARNING / CRITICAL.
   - Include events like:
     - Jetson heartbeat OK
     - Stream connected
     - Weapon model loaded
     - mmWave object extracted
     - Fallback mode activated
     - Server recovered
     - Alert acknowledged

13. Failure / Recovery UI
   - Show when server is down.
   - Show when Jetson switches to fallback mode.
   - Show recovery progress.
   - Show “Central mode restored” after recovery.
   - Display warnings without making the UI unusable.

DATA MODEL / MOCK DATA

Use mock data for now. The UI must be ready to connect to a FastAPI backend later.

Create mock objects for:

- devices
- camera streams
- thermal streams
- mmWave points
- detected persons
- detected weapons
- tracking IDs
- alerts
- system metrics
- heartbeat status
- mode status
- logs
- configuration

Example states:
- NORMAL_MODE
- FALLBACK_MODE
- FAILURE_DETECTED
- RECOVERY_MODE
- FAULT_MODE

Example device:
{
  id: "jetson-01",
  name: "Jetson 1",
  location: "Main Entrance",
  status: "online",
  heartbeat: "OK",
  rgbCamera: "online",
  thermalCamera: "online",
  mmwave: "online",
  fps: 28,
  latencyMs: 72,
  mode: "central"
}

Example alert:
{
  id: "alert-001",
  severity: "critical",
  type: "weapon_detected",
  source: "RGB + mmWave Fusion",
  deviceId: "jetson-01",
  confidence: 0.91,
  timestamp: "2026-05-20 14:32:18",
  trackingId: "person-07",
  status: "active"
}

SPRINT CONTEXT

The UI should support these upcoming development tasks:

Sprint 2.3:
- UI base setup.
- Live camera feed component.
- Detection overlay UI.
- Device listing UI.

Sprint 2.4:
- Alert UI.
- Weapon detection indicators.
- Metrics panel.
- mmWave overlay integration.

Sprint 2.5:
- Enhanced trajectory preview.
- Device health indicators.
- Mode controls.
- Failure state UI.
- Auto-refresh and reconnection logic.

Sprint 2.6:
- Recovery indicators.
- Alert visualization improvements.
- Config panel UI.

Sprint 2.7:
- Tracking visualization with IDs and paths.
- Path rendering using lines/trails.
- Unsafe person highlight UI.

Sprint 2.8:
- Full UI integration test.
- Failure simulation UI.
- Recovery flow testing.
- Demo polish with Ring-style UX.

TECH REQUIREMENTS

Use:
- React
- TypeScript
- Tailwind CSS
- Modular component architecture
- Mock data first
- Responsive desktop-first layout
- Clean reusable components
- No backend required yet
- No real authentication required yet
- No external paid UI libraries

Preferred components:
- DashboardLayout
- HeaderStatusBar
- DeviceGrid
- DeviceCard
- SensorPanel
- CameraFeedPanel
- ThermalFeedPanel
- MmWavePointCloudPanel
- DetectionOverlay
- AlertCenter
- MetricsPanel
- HealthMonitor
- ControlPanel
- ConfigPanel
- ConsoleLog
- ModeStatusBadge
- ThreatLevelIndicator

INTERACTION REQUIREMENTS

The UI should include simulated interactions:

- Select active Jetson device.
- Toggle overlays on/off.
- Toggle tracking paths.
- Simulate weapon alert.
- Simulate server failure.
- Simulate fallback mode.
- Simulate recovery.
- Change confidence threshold.
- Acknowledge alert.
- Clear alert.
- Switch between camera streams.
- Expand/collapse side panels.

VISUAL PRIORITY

The first screen must immediately communicate:
- What the system sees.
- Which sensors are active.
- Whether there is a threat.
- Whether the Jetsons are healthy.
- Whether the system is in normal or fallback mode.

Do not bury critical information.

The user should be able to understand system state in 3 seconds.

LAYOUT DIRECTION

Use this layout:

Top header:
- SCAN-U
- System Mode
- Threat Level
- Server Status
- Active Alerts
- Time

Left sidebar:
- Device list / CCTV-style grid
- Jetson health summary

Main center:
- Large 3-panel sensor area:
  1. RGB Camera
  2. Thermal Camera
  3. mmWave Point Cloud

Right sidebar:
- Alert Center
- AI confidence
- Sensor fusion status

Bottom dock:
- Metrics
- Console log
- Controls
- Config shortcut

Make the layout clean, not overloaded.

IMPORTANT SAFETY / PRODUCT POSITIONING

The UI is for detection, alerting, prevention, and situational awareness. It must not look like an offensive weapon system. Avoid militaristic targeting language like “engage target.” Use responsible safety terms:
- “Detected risk”
- “Potential weapon”
- “Unsafe person”
- “Alert”
- “Confidence”
- “Escalate”
- “Acknowledge”
- “Sensor fusion”
- “Presence detected”

DELIVERABLE

Create a complete frontend UI prototype.

Include:
- Project structure.
- React components.
- Tailwind styling.
- Mock data.
- Simulated mmWave point cloud.
- Simulated camera and thermal panels.
- Alert and system status flows.
- Clean, modern UI.
- Comments explaining where future FastAPI/WebSocket integration will connect.

The result should be visually impressive, investor-demo ready, and technically aligned with the SCAN-U architecture.


anade los endpoint del back end