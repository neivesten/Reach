# UR Robot Reach Training 

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0-orange)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3.10/)

## Demo

![UR Robot Reach Task](recordings/output.gif)

*The UR robot learning to reach target poses*

## Overview

This repository contains training code for our specific configured robot, Universal Robot (UR) robotic arm with Robotiq 2F-140 gripper, which is based on the NVIDIA Deep Learning Institute (DLI) course: **[Train Your Second Robot in Isaac Lab](https://www.nvidia.com/en-us/learn/learning-path/robotics/)**. The project implements a reinforcement learning pipeline where the robot learns to reach randomly sampled target poses within its workspace. The training uses manager-based RL environment architecture from Isaac Lab.


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). **Please use Isaac Lab with IsaacSim 4.5.0**.

- Clone this repository separately from the Isaac Lab installation:

```bash
git clone <repository-ssh>
cd Reach
```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode:

```bash
python -m pip install -e source/Reach

```

- Verify the installation by listing available tasks:

```bash
python scripts/list_envs.py
```

## Usage

### Training

Train a PPO agent on the reach task using SKRL:

```bash
python scripts/skrl/train.py --task Template-Reach-v0 --headless

```

### Testing

Run a trained policy for evaluation:

```bash
python scripts/skrl/play.py --task Template-Reach-Play-v0

```


## Environment Details

### Task Specification
- **Objective**: Reach target end-effector poses (position + orientation)
- **Observations**: Joint positions, velocities, pose commands, and previous actions
- **Actions**: Joint position commands (6-DOF for arm)
- **Reward Components**:
  - Position tracking error (L2 distance to target)
  - Orientation tracking error (quaternion difference)
  - Action rate penalty (smoothness)
  - Joint velocity penalty

### Training Configuration
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network Architecture**: 64x64 fully connected layers with ELU activation
- **Training Steps**: 24,000 timesteps by default
- **Environments**: 2000 parallel environments (training), 50 (play)

## Resources

For a comprehensive walkthrough of this implementation, watch the collaborative tutorial series by LycheeAI and I:

[![YouTube Tutorial: Deep Dive into UR Robot Training](https://img.youtube.com/vi/32uzEGpvSog/maxresdefault.jpg)](https://www.youtube.com/watch?v=32uzEGpvSog&t=44s&ab_channel=LycheeAI)


