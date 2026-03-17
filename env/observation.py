"""
Observation logic for MARL SGAT. This module defines how agents perceive the state of their
controlled intersection and the global network, providing the input for the Deep Q-Network.
"""
import traci
import numpy as np

class MARLObservation:
    """
    Manages the extraction of observations for a single traffic light agent.
    """
    def __init__(self, sumo_id):
        """
        Initializes the observation object for a specific SUMO traffic light ID.

        Args:
            sumo_id (str): The ID of the traffic light in the SUMO simulation.
        """
        self.sumo_id = sumo_id
        # Get all lanes controlled by this traffic light. This includes incoming and outgoing lanes.
        self.lanes = traci.trafficlight.getControlledLanes(sumo_id)
        # Extract unique lanes. Sometimes a lane appears multiple times if it's controlled by
        # multiple phases or for different turning movements. We only care about distinct physical lanes.
        self.unique_lanes = list(dict.fromkeys(self.lanes))

    def get(self):
        """
        Generates the observation vector for the agent associated with this SUMO ID.
        The observation vector is a 7-dimensional array designed to capture local traffic conditions
        and a global network metric, as described in the MARL SGAT paper.

        The structure is:
        [Density_L1, Queue_L1, ExitSpace_L1, Density_L2, Queue_L2, ExitSpace_L2, GlobalMetric]

        Returns:
            np.array: A 7-dimensional numpy array representing the current state observation.
        """
        obs = []

        # 1. Local Metrics for the first 2 controlled lanes (typically the most impactful or main lanes)
        # The paper often simplifies observations to a few key approaches/lanes for each intersection.
        for lane in self.unique_lanes[:2]: # Assuming we only need metrics for the first two unique lanes
            length = traci.lane.getLength(lane) # Length of the current lane in meters
            veh_num = traci.lane.getLastStepVehicleNumber(lane) # Number of vehicles on this lane in the last simulation step

            # Density: Number of vehicles per unit length of the lane.
            # A higher density indicates more congestion.
            obs.append(veh_num / length)

            # Queuing Density: Number of halting vehicles (speed < 0.1 m/s) per unit length.
            # This specifically measures the extent of queues forming.
            obs.append(traci.lane.getLastStepHaltingNumber(lane) / length)

            # Remaining Exit Space: A measure of how much capacity is left on the lane.
            # This helps the agent understand if vehicles can move into or out of the intersection effectively.
            # max_cap is an approximation of the maximum number of vehicles a lane can hold, assuming ~7.5m per vehicle.
            max_cap = length / 7.5
            obs.append(max(0, 1 - (veh_num / max_cap))) # Clamped between 0 and 1.

        # 2. Global Network Metric (Approximated by total expected vehicles)
        # This metric provides a coarse-grained overview of the overall network traffic pressure.
        # 'getMinExpectedNumber()' returns the total number of vehicles currently in the network
        # that are expected to arrive at their destinations (i.e., not yet departed or have departed and not arrived).
        # Divided by 100.0 for normalization to keep it within a reasonable range for neural network input.
        n_total = traci.simulation.getMinExpectedNumber()
        obs.append(n_total / 100.0)

        # Padding: Ensure the observation vector always has a fixed size (7 in this case).
        # If an intersection controls fewer than 2 relevant lanes or if the global metric isn't included
        # for some reason, pad with zeros. This maintains a consistent input shape for the DQN.
        while len(obs) < 7: obs.append(0.0)
        return np.array(obs, dtype=np.float32)
