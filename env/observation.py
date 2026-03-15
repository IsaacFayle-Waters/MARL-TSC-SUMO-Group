"""
Observation logic for MARL SGAT.
Calculates local lane metrics and global network inflow/outflow.
"""
import traci
import numpy as np

class MARLObservation:
    def __init__(self, sumo_id):
        self.sumo_id = sumo_id
        self.lanes = traci.trafficlight.getControlledLanes(sumo_id)
        self.unique_lanes = list(dict.fromkeys(self.lanes))

    def get(self):
        """
        Returns a 7-dimensional observation vector:
        [Density_L1, Queue_L1, ExitSpace_L1, Density_L2, Queue_L2, ExitSpace_L2, GlobalMetric]
        """
        obs = []
        # 1. Local Metrics for the first 2 controlled lanes
        for lane in self.unique_lanes[:2]:
            length = traci.lane.getLength(lane)
            veh_num = traci.lane.getLastStepVehicleNumber(lane)
            
            # Density
            obs.append(veh_num / length)
            # Queuing Density
            obs.append(traci.lane.getLastStepHaltingNumber(lane) / length)
            # Remaining Exit Space
            max_cap = length / 7.5
            obs.append(max(0, 1 - (veh_num / max_cap)))

        # 2. Global Network Metric (Approximated by total expected vehicles)
        n_total = traci.simulation.getMinExpectedNumber()
        obs.append(n_total / 100.0)

        # Padding to ensure fixed size of 7
        while len(obs) < 7: obs.append(0.0)
        return np.array(obs, dtype=np.float32)
