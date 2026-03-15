# MARL-TSC-SUMO-Group

## Project Purpose
This project is an experiment reproduction of the **'Multi-agent Deep Reinforcement Learning collaborative Traffic Signal Control method considering intersection heterogeneity'** (MARL SGAT).

## Objective
The goal is to maximize long-term cumulative reward (minimizing delay and maximizing throughput) across a road network using a 5x5 grid of 25 intersections controlled by decentralized DQN agents.

## Usage Instructions
1. **Environment Setup**: Open the `experiment_reprod_bie_et_al.ipynb` notebook in Google Colab.
2. **Initialization**: Run the first cell to clone this repository and install dependencies (SUMO, PettingZoo, etc.).
3. **Network Generation**: Execute the traffic network generation cells to build the grid and traffic flows.
4. **Training**: Run the training loop cell to begin the DQN optimization. The notebook is pre-configured with the MARL SGAT hyperparameters.
5. **Evaluation**: Use the provided evaluation cells to visualize agent actions and calculate average vehicle delay.

## Current Status
- Environment: PettingZoo-compatible wrapper for SUMO with Phase Duration Adjustment.
- Agent: Deep Q-Network (DQN) with Replay Buffer and SGAT Hyperparameters.
- Metrics: Average vehicle delay, density, queuing density, and throughput proxy.
