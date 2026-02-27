# Panorama + HRZ Field Test Instructions

## Goal

Figure out whether `GPSImgDirection` (or other EXIF/XMP metadata) can be used to reliably determine azimuth alignment for iPhone panoramas when generating N.I.N.A `.hrz` horizon files.

## What To Do (Next Test Session)

1. Take a new panorama from the exact same observing spot.
2. Immediately create a new N.I.N.A `.hrz` file from the same spot.
3. Save the original panorama file (`.HEIC`) and any `.xmp` sidecar file.
4. Record how the pano was captured (start facing direction, sweep direction).

## Recommended Capture Plan (High Value)

Do multiple panos back-to-back without moving your feet:

1. `2-4` panos from the exact same spot.
2. One sweep `left-to-right`.
3. One sweep `right-to-left`.
4. Start one pano facing `South` and another facing `East` (or another known direction).
5. Create an `.hrz` right after each pano.

This helps determine whether `GPSImgDirection` corresponds to:

- pano center
- start direction
- seam/anchor direction
- a consistent device-specific offset

## Metadata To Keep (Important)

For each pano, keep:

1. Original `.HEIC`
2. `.xmp` sidecar (if created)
3. EXIF/XMP values:
   - `GPSImgDirection`
   - `GPSImgDirectionRef` (`T` or `M`)
   - timestamp
   - GPS lat/lon
4. Note of sweep direction (`left->right` or `right->left`)
5. Note of start facing direction (for example: `started facing South`)

## Scene Markers (Best Improvement)

Add visible azimuth references in the scene:

1. Minimum: one known landmark marker for `North`
2. Better: two markers (`N` and `E`)
3. Best: `N`, `E`, `S`, `W` markers visible in the pano

Temporary cones/poles/tape are fine if you know their bearings.

## Compass / Capture Hygiene

To reduce heading errors:

1. Keep the phone away from metal equipment/tripods/mounts when starting the pano.
2. Check the iPhone Compass app before capture.
3. Note whether the phone is using `True North` or `Magnetic North` (if visible).
4. Try one capture handheld and one on a tripod (to test compass bias).

## Optional High-Confidence Validation

Take at least one pano with a visible Sun/sunset glow (or Moon) if possible.

Why: we can independently compute the Sun/Moon azimuth from:

- timestamp
- GPS location

and compare it to where it appears in the panorama.

## Suggested Notes Template (Per Pano)

Copy this for each capture:

```text
Pano ID / filename:
Time (local):
Spot: (same exact spot? yes/no)
Sweep direction: left->right / right->left
Start facing direction (estimated):
Visible markers in frame: N / E / S / W / other
Created matching HRZ file: yes/no (path)
Notes:
```

## What To Send Back For Analysis

For each test pano:

1. `.HEIC`
2. `.xmp` (if present)
3. matching `.hrz`
4. your notes (sweep direction + start facing direction)

With that set, we can test whether `GPSImgDirection` is:

- directly usable
- usable with a consistent offset
- dependent on sweep direction/start direction
- not reliable enough without a manual "click North" calibration step
