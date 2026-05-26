# Experiment 04: Robot Kinematics & Dynamics Analysis

## Results

![Analysis](robotics_analysis.png)

## Analysis Summary

### Franka Panda (7-DoF) — Kinematics

| Module | Method | Result |
|--------|--------|--------|
| Forward Kinematics (FK) | DH Parameters | End-effector: [0.484, 0, 0.413] m |
| Inverse Kinematics (IK) | LM Numerical | Position error: 0.000 mm |
| Jacobian Matrix | Geometric 6×7 | Condition number: 8.91 |
| Trajectory Planning | 5th-order Polynomial | 50-step smooth trajectory |
| Manipulability | Yoshikawa Index | Home config: 0.0838 |

### Puma560 (6-DoF) — Dynamics

| Module | Method | Result |
|--------|--------|--------|
| Gravity Compensation | gravload | [0, -0.775, 0.249, 0, 0, 0] N·m |
| Inertia Matrix | inertia | 6×6, diagonal: [2.34, 5.21, 0.94, ...] |
| Inverse Dynamics | rne (Newton-Euler) | Static torques verified |

## Key Findings

1. **IK accuracy**: LM solver achieves 0.000 mm position error on Franka Panda
2. **Jacobian condition number = 8.91**: well away from singularity (threshold ~100)
3. **Manipulability = 0.0838**: consistent along trajectory, no singularity risk
4. **Gravity dominates dynamics**: joints 2-3 carry significant gravity load

## Why Two Robots?

- **Franka Panda**: modern 7-DoF robot, excellent kinematics model, no dynamics params
- **Puma560**: classic 6-DoF robot, complete dynamics parameters available
- Together they demonstrate the full pipeline: FK → IK → Jacobian → Trajectory → Dynamics

## Connection to Isaac Lab Experiments

| Dimension | This Experiment | Isaac Lab (Exp02 Allegro) |
|-----------|----------------|--------------------------|
| Tool | Robotics Toolbox | PhysX GPU |
| Speed | Microseconds | Milliseconds |
| Purpose | Algorithm analysis | Large-scale RL training |
| Focus | Kinematics/Dynamics | Policy learning |

## Dependencies 
roboticstoolbox-python==1.1.1
spatialmath-python
numpy<2.0
scipy
matplotlib
