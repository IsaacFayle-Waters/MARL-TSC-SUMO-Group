# MARL-TSC-SUMO-Group

## Project Purpose
This project is an experiment reproduction of the **'Multi-agent Deep Reinforcement Learning collaborative Traffic Signal Control method considering intersection heterogeneity'** (MARL SGAT).

## Objective
The goal is to maximize long-term cumulative reward (minimizing delay and maximizing throughput) across a road network using a 5x5 grid of 25 intersections controlled by decentralized DQN agents.

## Key Features
- **Intersection Heterogeneity**: Incorporates a 'Heterogeneous Correlation Index' into the reward function to account for structural differences and relationships between intersections.
- **Phase Duration Adjustment**: Agents adjust green light durations (+4s / -4s) at 12-second reassessment intervals, rather than simple discrete switching.
- **Dynamic Parsing**: The environment automatically extracts physical lane counts and distances from the SUMO `network.net.xml` file.

## Usage Instructions
1. **Environment Setup**: Open the `experiment_reprod_bie_et_al.ipynb` notebook in Google Colab.
2. **Initialization**: Run the setup cells to clone this repository and install SUMO and PettingZoo dependencies.
3. **Network Generation**: Generate the 5x5 grid and high-density traffic flows (5000 trips) using the provided SUMO utility cells.
4. **Training**: Execute the training loop configured with MARL SGAT hyperparameters: `gamma=0.9`, `batch_size=32`, and `q_scaling=3.0`.

## Current Status
- **Environment**: Fully aligned with MARL SGAT Section 4.1.
- **Scale**: 25 agents (5x5 grid).
- **Metrics**: Integrated average delay, throughput, and spatiotemporal correlation rewards.
