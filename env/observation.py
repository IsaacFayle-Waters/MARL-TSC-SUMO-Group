import traci

def lane_density(lane):
    vehicles = traci.lane.getLastStepVehicleNumber(lane)
    length = traci.lane.getLength(lane)
    return vehicles / length
