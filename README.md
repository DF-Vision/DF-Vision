# Dual-Signal Calibration for Robust Robotic Manipulation

A robotics project focused on improving the reliability of robotic grasping by combining visibility estimation, boundary-aware scene understanding, conformal prediction, and natural voice interaction.

---

## Overview

Robotic grasping often fails because the robot's perception of the environment differs from reality.

This project combines two complementary signals:

- **Calibration Gap (CG)** – Measures the difference between perceived and actual object visibility.
- **Physical Relation Graph (PRG) Edge Error** – Measures how accurately object boundaries and spatial relationships are perceived.

These signals are fused into a unified framework to predict grasp failures before execution, allowing the robot to safely defer uncertain actions while providing grounded spoken explanations.

---

## Features

- YOLOv8 Multi-Object Detection
- RGB-D Perception
- TSDF-based Visibility Estimation
- Physical Relation Graph Integration
- Boundary-aware Depth Processing
- Dual-Signal Failure Prediction
- Conformal Safety Gate
- Voice Command Support
- Grounded Spoken Explanations
- Comprehensive Experiment Logging

---

## Pipeline

```
Voice Command
      │
      ▼
Speech Recognition
      │
      ▼
Object Grounding
      │
      ▼
RGB-D Perception
      │
      ▼
YOLO Detection + PRG Feature Extraction
      │
      ▼
Calibration Gap Computation
      │
      ▼
Dual-Signal Failure Prediction
      │
      ▼
Conformal Safety Gate
      │
      ▼
Execute Grasp / Defer Action
      │
      ▼
Grounded Spoken Explanation
```

---

## Tech Stack

- Python
- ROS2 Humble
- PyTorch
- YOLOv8
- OpenCV
- Open3D
- Whisper
- pyttsx3
- NumPy
- Pandas
- Scikit-learn

---

## Hardware

- SO-101 Robot Arm
- Intel RealSense D435i / Orbbec Gemini 2
- RGB Camera
- Microphone
- Speakers

---

## Repository Structure

```
.
├── calibration_gap/
├── prg/
├── perception/
├── conformal/
├── voice/
├── hardware/
├── experiments/
├── logger/
├── scripts/
├── configs/
├── docs/
└── README.md
```

---

## Current Status

- ✅ Hardware Pipeline
- ✅ Object Detection
- ✅ Calibration Gap Computation
- ✅ PRG Integration
- ✅ Conformal Safety Gate
- ✅ Voice Interaction
- 🚧 Ongoing Development & Evaluation

---

## Contributors

<table>
<tr>
<td align="center">
<a href="https://github.com/krupa-shah23">
<img src="https://github.com/krupa-shah23.png" width="100px;" alt="Krupa Shah"/><br />
<b>Krupa Shah</b>
</a>
<br />
<sub>@krupa-shah23</sub>
</td>

<td align="center">
<a href="https://github.com/manasi-coding">
<img src="https://github.com/manasi-coding.png" width="100px;" alt="Manasi Sawant"/><br />
<b>Manasi Sawant</b>
</a>
<br />
<sub>@manasi-coding</sub>
</td>
</tr>
</table>

---

## License

MIT License
