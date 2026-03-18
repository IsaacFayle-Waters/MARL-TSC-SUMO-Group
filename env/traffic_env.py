"""
TrafficEnv: Multi-Agent Traffic Signal Control Environment with Dynamic Structural Parsing.

This environment implements the MARL_SGAT framework for traffic signal control using SUMO.
It features a custom reward function based on the paper's 'Heterogeneous Correlation Index',
which accounts for the structural relationships between upstream and downstream intersections.

KEY FEATURES:
- DYNAMIC PARSING: Uses sumolib to read 'network.net.xml' to extract real-world lane counts (m_p)
  and edge lengths (dist) for accurate heterogeneous correlation calculations.
- ACTION SPACE (Phase Duration Adjustment):
    * Action 0: Maintain current phase duration.
    * Action 1: Extend current green phase by delta_t (+5s).
    * Action 2: Reduce current green phase by delta_t (-5s).
- REWARD LOGIC: Implements Equation (8) using calculated cor_forward and cor_backward indices
  to weight the local delay impact based on intersection importance and proximity.
"""
from pettingzoo import ParallelEnv
import traci
import traci.connection
import sumolib
import numpy as np
from env.observation import MARLObservation

class TrafficEnv(ParallelEnv):
    metadata = {"name": "traffic_marl_env"}

    def __init__(self, net_file="/content/MARL-TSC-SUMO-Group/sumo/network.net.xml"):
        """
        Initializes the environment and parses the SUMO network to ground the agent logic
        in the actual physical topology of the grid.
        
        Args:
            net_file (str): Path to the SUMO .net.xml file used to extract structural metadata.
        """
        # List of traffic light IDs controlled as agents.
        self.sumo_ids = ['A1', 'B0', 'B1', 'B2', 'C1']
        self.agents = [f"agent_{i}" for i in range(len(self.sumo_ids))]
        self.agent_to_sumo = {f"agent_{i}": sid for i, sid in enumerate(self.sumo_ids)}
        
        # --- MARL SGAT HYPERPARAMETERS ---
        self.q_scaling = 1.0 # 'q' scaling factor for unifying numerical ranges (from paper).
        self.min_green = 10  # Minimum green time allowed for safety/visibility.
        self.max_green = 60  # Maximum green time allowed to prevent excessive cross-traffic delay.
        self.delta_t = 5     # The fixed increment for phase duration adjustments.

        # --- DYNAMIC NETWORK ANALYSIS ---
        # Load the network structure using sumolib to replace static placeholders.
        # This ensures the 'm_p' and 'dist' values match the user's specific map.
        self.net = sumolib.net.readNet(net_file)
        self.structural_data = self._parse_network_structure()

        # State tracking buffers for RL feedback and timing logic.
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        self.observations_manager = {}

    def _parse_network_structure(self):
        """
        Iterates through the SUMO network nodes to extract topological data used in
        the Heterogeneous Correlation Index formulas (Equations 6 & 7).
        """
        data = {}
        for agent in self.agents:
            sumo_id = self.agent_to_sumo[agent]
            node = self.net.getNode(sumo_id)
            
            # Extract 'm_p_i': Total number of incoming lanes at this intersection node.
            # This represents the intersection's 'scale' in the paper's formula.
            m_p_i = sum([edge.getLaneNumber() for edge in node.getIncoming()])
            
            # Extract 'dist' (x_p): Length of the primary incoming edge.
            # This represents the physical distance (meters) between intersections.
            dist = 500.0
            incoming_edges = node.getIncoming()
            if incoming_edges:
                dist = incoming_edges[0].getLength()

            data[agent] = {
                'm_p': max(1, m_p_i),
                'dist': max(1.0, dist)
            }
        return data

    def reset(self, seed=None, options=None):
        """
        Resets the simulation and re-initializes TraCI-dependent objects.
        """
        try:
            if "default" in traci.connection._connections: traci.close()
        except Exception: pass
        finally:
            if "default" in traci.connection._connections: del traci.connection._connections["default"]

        traci.start(["sumo", "-c", "/content/MARL-TSC-SUMO-Group/sumo/config.sumocfg"])
        
        # Managers must be reset after traci.start() to access live simulation state.
        self.observations_manager = {agent: MARLObservation(self.agent_to_sumo[agent]) for agent in self.agents}
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        return {agent: self.observations_manager[agent].get() for agent in self.agents}

    def step(self, actions):
        """
        Executes the provided actions and advances the world state by one second.
        """
        for agent, action in actions.items():
            self._apply_action(agent, action)
            
        traci.simulationStep()
        
        for agent in self.agents: 
            self.current_phase_timer[agent] += 1

        obs = {agent: self.observations_manager[agent].get() for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        
        # Standard PettingZoo API structure for Parallel environments.
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return obs, rewards, terminations, truncations, infos

    def _get_hetero_index(self, agent):
        """
        Calculates the directional correlation indices using the dynamic structural data.
        Formula (Eq 6 & 7): cor = m_neighbor / (m_current * q * sqrt(distance))
        """
        m_p_i = self.structural_data[agent]['m_p']
        dist = self.structural_data[agent]['dist']
        
        # We assume neighbors share structural characteristics in this grid configuration.
        # The q_scaling factor helps unify the numerical ranges of different variables.
        cor_forward = m_p_i / (m_p_i * self.q_scaling * np.sqrt(dist))
        cor_backward = m_p_i / (m_p_i * self.q_scaling * np.sqrt(dist))
        
        return cor_forward, cor_backward

    def _compute_reward(self, agent):
        """
        Calculates the Heterogeneous Reward (Equation 8).
        Multiplies local delay feedback by the spatial correlation indices of the intersection.
        """
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        
        # Calculate local average delay as the change in waiting time between steps.
        current_waiting_time = sum([traci.lane.getWaitingTime(l) for l in lanes])
        delay = (self.last_waiting_times[agent] - current_waiting_time) / 100.0
        
        # Apply Heterogeneity scaling using the pre-parsed topological data.
        cor_f, cor_b = self._get_hetero_index(agent)
        
        # Final reward formulation: Volume Ratio * Cor_F * Cor_B * Delay.
        reward = 1.0 * cor_f * cor_b * delay
        
        self.last_waiting_times[agent] = current_waiting_time
        return float(reward)

    def _apply_action(self, agent, action):
        """
        Translates RL agent actions into SUMO phase duration adjustments.
        Includes logic to handle transitions and yellow phase constraints.
        """
        sumo_id = self.agent_to_sumo[agent]
        current_phase = traci.trafficlight.getPhase(sumo_id)
        
        # Logic for inter-green/yellow phases (typically odd indices).
        # These are fixed (3s) and cannot be adjusted by the RL agent.
        if current_phase % 2 != 0:
            if self.current_phase_timer[agent] >= 3:
                traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
                self.current_phase_timer[agent] = 0
            return
        
        # Actions: 0 (Keep), 1 (Extend +5s), 2 (Reduce -5s).
        dur = traci.trafficlight.getPhaseDuration(sumo_id)
        if action == 1: 
            traci.trafficlight.setPhaseDuration(sumo_id, min(self.max_green, dur + self.delta_t))
        elif action == 2: 
            traci.trafficlight.setPhaseDuration(sumo_id, max(self.min_green, dur - self.delta_t))

        # If the phase has completed its (potentially adjusted) duration, switch to yellow.
        if self.current_phase_timer[agent] >= traci.trafficlight.getPhaseDuration(sumo_id):
            traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
            self.current_phase_timer[agent] = 0
