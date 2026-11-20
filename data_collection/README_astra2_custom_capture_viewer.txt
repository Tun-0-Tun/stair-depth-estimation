ASTRA 2 CUSTOM CAPTURE VIEWER
============================

This application talks directly to the Orbbec Astra 2 with pyorbbecsdk2.

FIXED CAPTURE PROFILE
----------------------
Depth : 1600x1200 @ 15 FPS, Y16
Color : 1920x1080 @ 15 FPS, YUYV
Depth work mode: High Resolution
SDK depth post-processing: none
IMU: ACCEL + GYRO when exposed by the device/SDK

WHY THIS IS RESEARCH-SAFE
--------------------------
The application does not write filtered/aligned/normalized depth into the
bag. The depth preview is only a display conversion. RAW frames are written
through Orbbec's SDK-native RecordDevice interface.

The program refuses to silently substitute a different RGB or depth profile.
If an exact requested profile is not available, it prints the available
profiles and stops.

INSTALL
-------
Python 3.9+ recommended.

pip install pyorbbecsdk2 opencv-python numpy

On Windows, if the SDK documentation requires frame-metadata registration,
install the metadata support described in the pyorbbecsdk repository before
running the application.

RUN
---
python astra2_custom_capture_viewer.py

Optional:
python astra2_custom_capture_viewer.py --duration 1.5 --output recordings --prefix stair

CONTROLS
--------
SPACE   capture one independent short .bag burst
Q/ESC   quit

FILES
-----
For each capture:
    stair_000001_YYYYMMDD_HHMMSS_mmm.bag
    stair_000001_YYYYMMDD_HHMMSS_mmm.json

The JSON is a sidecar manifest describing the requested and actual stream
profiles, device information, capture timestamps, and observed pipeline frame
counters.

IMPORTANT FIRST TEST
--------------------
Before collecting the full dataset:
1. Start the viewer.
2. Verify the displayed profile is exactly 1600x1200 Y16 @ 15 and
   1920x1080 YUYV @ 15.
3. Verify depth mode shows High Resolution.
4. Verify ACCEL and GYRO are receiving data.
5. Press SPACE once.
6. Confirm both the .bag and .json appear.
7. Replay the bag using Orbbec's SDK playback example and verify that depth, color, and IMU data are present.
