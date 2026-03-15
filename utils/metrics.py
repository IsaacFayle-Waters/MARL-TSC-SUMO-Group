"""
Utility functions for calculating traffic performance metrics via TraCI.
Optimized for Reinforcement Learning feedback.
"""
import traci

def average_delay():
    """
    Calculates the average accumulated waiting time across all vehicles currently in the simulation.
    Using accumulatedWaitingTime is more robust for RL than getWaitingTime.
    """
    veh_ids = traci.vehicle.getIDList()
    if len(veh_ids) == 0:
        return 0.0

    total_accumulated_delay = 0.0
    for v in veh_ids:
        # getAccumulatedWaitingTime tracks total time spent at speed < 0.1m/s
        total_accumulated_delay += traci.vehicle.getAccumulatedWaitingTime(v)

    return total_accumulated_delay / len(veh_ids)
