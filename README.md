# Multi-Agent Trajectory Optimization with Decentralized NMPC

This repository contains the implementation of a decentralized Nonlinear Model Predictive Control (NMPC) approach for multi-agent trajectory optimization, with a focus on parallel computing techniques to enhance performance.

## Overview

Our work extends the multi-agent path planning framework by Ashwin Bose, incorporating GPU-accelerated algorithms for obstacle prediction and collision cost computation, as well as a custom Karush-Kuhn-Tucker (KKT) solver for trajectory optimization.

Key features:

- Decentralized NMPC formulation for multi-agent trajectory optimization with dynamic collision avoidance
- GPU-accelerated algorithms for obstacle prediction and collision cost computation
- Custom KKT solver for trajectory optimization
- Parallel implementation using CUDA

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/multi-agent-trajopt.git
   cd multi-agent-trajopt
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Ensure you are using a CUDA-compatible GPU and have the necessary drivers installed.

## Usage

To run the multi-agent trajectory optimization:

```
cd decentralized
!python decentralized.py -m nmpc -f test.gif
```

This will run the decentralized NMPC algorithm and save the output visualization as `test.gif`.

## Team

- Adheesh Kadiresan
- Arata Katayama
- Joseph Nicol
- Brandon Pae
- Taimur Shaikh

## Acknowledgements

Citations:
[1] https://github.com/atb033/multi_agent_path_planning
