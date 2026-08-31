# Dual AWR1843 + dual RGB on one Linux server

## Decision

This is an independent laboratory boundary for two facing AWR1843BOOST boards
and two USB RGB cameras connected directly to one Linux server. It intentionally
removes the Jetson, SSH transfer and Tailscale clocks from the acquisition data
path. It reuses the existing `lab.mmwave77_usb` capture/calibration/perception
code and `lab.dual_mmwave77_stereo` geometric fusion/rendering code.

It is not a fifth canonical SCAN-U workflow and its artifacts cannot enter the
schema-V6 training set. The internal module entrypoint is a bounded hardware
diagnostic authorized for this laboratory topology.

```text
AWR1843 A CLI/TLV ─┐
AWR1843 B CLI/TLV ─┤
Camera A V4L2 ─────┼─ one Linux host monotonic clock ─ local immutable spool
Camera B V4L2 ─────┘                                  │
                                                      ├─ per-radar cube/background/perception
                                                      ├─ B -> A facing transform
                                                      ├─ fused points + global tracks
                                                      └─ Camera A | fused radar | Camera B
```

## Acquisition

- XDS110 CLI/data pairs are resolved by USB topology and CLI/data interface
  identity; `/dev/ttyACM*` numbers are not configured permanently.
- Cameras prefer stable `/dev/v4l/by-id/*-video-index0` links. Distinct real
  devices are required.
- Four child processes drain the two UARTs and two compressed camera streams.
- Cameras begin before the radar processes and end after them. Every source
  preserves its own host-monotonic timeline.
- Raw UART, decoded TLVs, range/noise profiles and native camera MP4s are
  retained before derived rendering.

## Synchronization boundary

One host removes cross-computer clock offset, but USB receipt time remains an
observation timestamp rather than a hardware exposure/chirp timestamp. Radar
TI cycle counters and camera frame timing are retained. The fusion matcher
records actual pairing deltas and fails when no windows meet the configured
tolerance. No RF phase coherence is claimed.

## Geometry and tracking

The compatibility transform for exactly facing, equal-height radars is
`x'=-x, y'=D-y, z'=z`. `D` must be measured antenna-to-antenna. Residual yaw,
pitch, roll and height errors remain a calibration limitation. Global tracks
deduplicate post-CFAR clusters; they do not turn two 12-virtual-channel devices
into one coherent 24-channel aperture.

## Safety boundary

Yellow/red reflectivity cues are uncalibrated engineering evidence. They do
not confirm material and do not classify a firearm. Cameras are visual context;
visible-firearm evidence requires a separately validated trained checkpoint.

