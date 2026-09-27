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

**Correction (2026-09-27, `fix/readme-status-correction`): the status
list this section previously showed (dated 2026-07-30, this repo's first
week) claimed "✅ PRG Integration," "✅ Hardware Pipeline," and
"✅ Voice Interaction" as complete. None of that was accurate then and
none of it is accurate now — corrected below rather than deleted, so
this file doesn't silently go stale the same way twice.**

This section no longer restates status inline — status changes too
often and inline copies drift out of date (exactly what happened here).
**For current, authoritative status, read
`project_final/PROJECT_STATUS_FINAL.md` on the `project/final-integration`
branch** (not present on this branch — this
correction branch was created directly from the old `failure-calibration`
branch, per this project's own discipline of never touching another
branch's history; check out `project/final-integration` to read it). As
of that document (2026-09-27):

- **Software**: every planned analysis path (Gap-only C1/C4, the
  now-retired PRG dual-signal extension, Model D) is built, tested, and
  — where a safety/statistics guard exists — mutation-hardened. This
  part is genuinely done.
- **PRG dual-signal integration (C1-ext, C4-ext, Model D)**: NOT
  complete, NOT working. Five sessions, three distinct algorithmic
  designs, all failed decisively against pre-committed evaluation
  criteria. Formally **retired** for this paper's submission (dated
  amendment in `perception-calibration-gap/PREREG.md`); Gap-only is the
  sole primary path.
- **Hardware pipeline**: infrastructure (calibration solver, grasp
  lookup table, YOLO training loop, Checkpoint 1 validation tooling) is
  built and tested on synthetic data only. **Zero real trials have ever
  been run.** Blocked on physical hardware access, not on more software.
- **Voice interaction**: never started, per this project's own planning
  documents.

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
