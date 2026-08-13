# AWR1843 human-centric perception methodology

## Scope

This laboratory workflow studies whether processed AWR1843BOOST USB TLVs can
support understandable human tracking and body-associated reflective-anomaly
evidence. It is not a validated metal identifier or firearm detector. Only
controlled inert objects are used.

## Evidence boundary

The XDS110 Auxiliary UART provides post-CFAR detected points, side information,
range/noise profiles and processing statistics. It does not provide complex
virtual-antenna ADC samples. Empty point-cloud voxels therefore mean “no demo
detection was reported,” not proven empty physical space.

The interpretation chain is:

```text
byte-exact UART
  → decoded points and complete profiles
  → transport/profile quality
  → empty-room baseline and clutter mask
  → spatial clusters
  → temporal human track
  → track-relative compact reflectors
  → auditable anomaly components
```

## Required controlled conditions

Collect matched sessions in the same sensor pose and room:

1. `empty_room`
2. `person_only`
3. `person_with_inert_test_object`

Use the same participant, clothing, start position, path and timing for
conditions 2 and 3. Repeat at 1.5, 2.0, 2.5 and 3.0 m. A useful 60-second
sequence is:

- 10 seconds empty;
- 15 seconds front-facing;
- 10 seconds left profile;
- 10 seconds right profile;
- 10 seconds slow turn;
- 5 seconds exit.

Record stable participant ID, measured distance, body-relative object
location, room, day, clothing and object identity. Include person-only and
benign hard negatives such as a phone, keys, belt buckle, laptop, water bottle
and tool. Labels come from the controlled setup, never the radar output.

## Baseline and quality

The empty-room model uses median, MAD, 5th/95th percentiles and a bounded
robust scale independently for every range-profile bin. Noise profiles remain
a separate recorded reference. Sparse RAE occupancy marks the near field and
stable clutter voxels. No fixed inverse-range correction replaces the measured
baseline.

A calibration is invalid after moving the radar or major scene geometry. A
reset without physical movement does not change the stored geometry, but a new
empty-room capture is recommended for a new screening session.

## Human track

Points are clustered in Cartesian space after near-field and calibrated
clutter suppression. Track creation requires a multi-point moving cluster.
Subsequent windows use a constant-velocity alpha-beta update and bounded
association gate. The displayed ellipsoid represents the observed radar
volume only; limited elevation aperture prevents anatomical reconstruction.

## Reflective-anomaly evidence

Only strong local returns inside the tracked volume are considered. Every
candidate exposes:

- empty-room range-profile robust residual;
- SNR above the local body context;
- spatial compactness;
- Doppler coupling with track motion;
- track-relative temporal persistence;
- measured center, extent and uncalibrated observed-volume region.

The weighted result is `reflective_anomaly_score`, an engineering ranking.
It is not a material probability. Without background residual, stable person
evidence and temporal persistence, screening remains `insufficient_signal`.
Even a qualifying result is only `suspicious_metal`; it is not confirmed metal
or a firearm classification.

## Evaluation

Compare `person_with_inert_test_object` primarily against its matched
`person_only` capture, with `empty_room` used for scene subtraction. Report:

- clutter suppression fraction;
- track continuity and distance error;
- anomalies per participant-minute;
- false positives for every benign object;
- sensitivity by distance and orientation;
- score distributions and capture-clustered confidence intervals;
- insufficient-signal rate.

Keep all windows from one participant/capture/distance group in one split.
Never split overlapping windows between training and evaluation. Begin with
documented logistic-regression or tree baselines and probability calibration;
use temporal or 3D networks only after they outperform that baseline on
participant-, object-, distance-, room- and day-held-out captures.

## Hardware escalation

If the research goal requires paper-like dense range–azimuth–elevation images,
move to LVDS/DCA1000 raw ADC and implement range, Doppler and calibrated
TDM-MIMO angle processing. Software interpolation of sparse TLV points cannot
recover the AWR1843 aperture information that the demo firmware did not emit.
