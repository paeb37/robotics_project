# Multi-Agent Trajectory Optimization with Decentralized NMPC

This repository contains the implementation of a decentralized Nonlinear Model Predictive Control (NMPC) approach for multi-agent trajectory optimization, with a focus on parallel computing techniques to enhance performance.

## Overview

Our work extends the multi-agent path planning framework by Ashwin Bose, incorporating GPU-accelerated algorithms for obstacle prediction and collision cost computation, as well as a custom Karush-Kuhn-Tucker (KKT) solver for trajectory optimization.

Key features:

- Decentralized NMPC formulation for multi-agent trajectory optimization with dynamic collision avoidance
- GPU-accelerated algorithms for obstacle prediction and collision cost computation
- Custom KKT solver for trajectory optimization
- Parallel implementation using CUDA

## Usage

1. Clone the repository and switch to the branch we are working on:

   ```
   git clone https://github.com/paeb37/robotics_project.git
   git checkout taimur/integrate-decentralized-parallel-cost
   ```

2. Run the notebook `decentralized/RoboticsTest.ipynb` on a machine that has a CUDA-compatible GPU (e.g. Google Colab using a T4) to run the simulation. This will run the decentralized NMPC algorithm and save the output visualization as `test.gif`.

## Team

- Adheesh Kadiresan
- Arata Katayama
- Joseph Nicol
- Brandon Pae
- Taimur Shaikh

## Acknowledgements

Citations:
[1] https://github.com/atb033/multi_agent_path_planning
