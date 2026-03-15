"""
Utility functions for calculating traffic performance metrics via TraCI.
"""
import traci

def average_delay():
    """
    Calculates the average waiting time (delay) across all vehicles currently in the simulation.
    Waiting time is defined by SUMO as speed < 0.1 m/s.
    """
    veh_ids = traci.vehicle.getIDList()

    if len(veh_ids) == 0:
        return 0

    total = 0
    for v in veh_ids:
        total += traci.vehicle.getWaitingTime(v)

    return total / len(veh_ids)
