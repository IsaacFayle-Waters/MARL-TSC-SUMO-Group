import traci

def average_delay():
    veh_ids = traci.vehicle.getIDList()

    if len(veh_ids) == 0:
        return 0

    total = 0
    for v in veh_ids:
        total += traci.vehicle.getWaitingTime(v)

    return total / len(veh_ids)
