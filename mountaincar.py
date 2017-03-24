import gym
from players.manager import Manager


def run_episode(env, manager, nn, i, count):
    observation = env.reset()
    nn.fitness = 0
    for step in range(300):

        # env.render()
        # print(env.action_space)

        action = manager.get_action(observation, nn)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        nn.fitness += reward

        if done:

            nn.fitness += 300

            if nn.is_champ:
                print("champion agent: {0}".format(nn.fitness_previous))
            print("agent: {0}, raw score: {1}, steps: {2}".format(i, nn.fitness, step))
            return

    nn.fitness += 300
    if nn.is_champ:
        print("champion agent: {0}".format(nn.fitness_previous))
    print("agent: {0}, raw score: {1}, steps: {2}".format(i, nn.fitness, 300))


def run():
    env = gym.make('MountainCar-v0')
    n_input = 2
    n_output = env.action_space.n
    n_nn = 250
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=2, c2=2, c3=1, bias=1, drop_rate=0.8)
    manager.initialize()

    count = 0
    for i_episode in range(200):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode(env, manager, nn, i, count)
            count += 1
        manager.create_next_generation()


if __name__ == "__main__":
    run()
