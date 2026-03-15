# MARL-TSC-SUMO-Group

## Project Purpose
This project is an experiment reproduction of the **'Multi-agent Deep Reinforcement Learning collaborative Traffic Signal Control method considering intersection heterogeneity'** (MARL SGAT).

## Objective
The goal is to maximize long-term cumulative reward (minimizing delay and maximizing throughput) across a road network using a 5x5 grid of 25 intersections controlled by decentralized DQN agents.

## Current Status
- Environment: PettingZoo-compatible wrapper for SUMO.
- Agent: Deep Q-Network (DQN) with Replay Buffer.
- Metrics: Average vehicle delay and throughput-based rewards.
