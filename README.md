# HoloGesture — Gesture-driven Image Transform Prototype

Lightweight prototype that maps hand gestures from a webcam to image transforms.  
Current stage: **2D image transforms** (zoom + rotation).
Next stage: integrate full **3D model control**.

---

## Features
- Real-time hand tracking using **MediaPipe Hands**  
- Finger gesture detection to trigger actions  
- **Pinch Zoom** (thumb + index)  
- **3-Finger, 4-Finger, 5-Finger** gestures for X, Y, Z rotation  
- **Closed fist** resets all transforms  
- Perspective warp for simulated 3D rotation

---

## Requirements
```bash
pip install opencv-python mediapipe numpy
```
---

## Gesture Mapping
Gesture	     Action	       Description
[1,1,0,0,0]	  Zoom	         Pinch thumb + index to zoom in/out
[1,1,1,0,0]	  Rotate X	     Tilt hand to rotate on X axis
[1,1,1,1,0]	  Rotate Y	     Tilt hand to rotate on Y axis
[1,1,1,1,1]	  Rotate Z	     Rotate hand to spin image (Z axis)
[0,0,0,0,0]	  Reset	         Reset zoom and rotation

## How it Works
* MediaPipe detects 21 hand landmarks per frame

* count_fingers() identifies which fingers are extended

* Zoom: Distance between thumb tip (4) and index tip (8)

* Rotation: Calculated from wrist–middle MCP vector angle

* Perspective warp (simulate_3d_rotation()) simulates X/Y tilts

* Affine transform (transform_image()) applies Z rotation and zoom

## Tuning
* Modify thresholds (diff > 2, abs(delta) > 1) for sensitivity

* Add smoothing filters for steadier control

* Clamp zoom_level between 0.5–2.5 for stability

## Next Steps
* Integrate a 3D renderer (OpenGL / moderngl)

* Map gestures to 3D camera transforms (pitch, yaw, roll)

* Load and manipulate .obj or .glb models

* Add UI for calibration and gesture remapping

## File Structure
```
📁 HoloGesture/
│
├── hologesture.py      # Main program file
├── sample.jpeg         # Image for gesture manipulation
└── README.md           # Project documentation
```
