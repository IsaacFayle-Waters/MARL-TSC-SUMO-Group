from env.traffic_env import TrafficEnv

def train():

    env = TrafficEnv()

    for episode in range(200):

        observations = env.reset()

        for step in range(300):

            actions = {agent: 0 for agent in observations}

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            observations = next_obs

if __name__ == "__main__":
    train()
