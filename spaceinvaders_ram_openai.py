import gym
from players.manager import Manager
import os
from gym import wrappers


def run_episode(env, manager, nn, i, n_epi=1, render=False, random=False):
    """
    runs each episode

    :param env: OpenAI environment
    :param manager: NEAT manager
    :param nn: neural network to run the episode
    :param i: neural network index
    :param n_epi: number of times to run episode
    :param render: if True, renders the game play
    :param random: if True, agent moves randomly
    """
    observation = env.reset()
    nn.fitness = 0
    fitness_sum = 0

    for epi in range(n_epi):
        for step in range(100000):

            if render:
                env.render()

            observation = observation/255
            if random:
                action = env.action_space.sample()
            else:
                action = manager.get_action(observation, nn)
            observation, reward, done, info = env.step(action)
            fitness_sum += reward

            if done:
                observation = env.reset()
                break

    nn.fitness = fitness_sum / float(n_epi)
    if nn.is_champ:
        print("champion agent: {0}".format(nn.fitness_previous))
    print("agent: {0}, raw score: {1}".format(i, nn.fitness))

    return nn.fitness


def run():
    """
    Runs NEAT training
    """

    n_input = 128
    n_output = env.action_space.n
    n_nn = 150
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=300, c2=300, c3=150, bias=1, drop_rate=0.8,
                      weight_max=3, weight_min=-3, weight_mutate_rate=0.01, pm_weight_random=0.2)
    manager.initialize()

    dir_path = os.path.join(os.getcwd(), "results")
    file_path = os.path.join(dir_path, "invaders_fitness_history_openai.txt")

    # run 40 generations
    for i_episode in range(40):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode(env, manager, nn, i, n_epi=3)

        # record best fitness
        manager.remember_best_nn()
        with open(file_path, "a") as f:
            f.write(str(manager.nn_best.fitness))
            f.write("\n")
        manager.create_next_generation()

    # record best neural network
    manager.write_best_nn("invaders_result_openai.txt")


def check_result():
    """
    Validates the best performed neural network
    """

    n_input = 128
    n_output = env.action_space.n
    n_nn = 1
    n_trials = 100
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=300, c2=300, c3=100, bias=1, drop_rate=0.8)
    nn = manager.recreate_best_nn("invaders_result.txt")

    # runs 100 times and see the results
    fitness_sum = 0
    for i_episode in range(n_trials):
        print("running episode: {0}".format(i_episode))
        fitness = run_episode(env, manager, nn, -1, n_epi=1, render=False, random=True)
        fitness_sum += fitness

    print(fitness_sum / float(n_trials))

if __name__ == "__main__":
    # initialize test environment
    env = gym.make('SpaceInvaders-ram-v0')
    env = wrappers.Monitor(env, 'tmp/spaceinvaders-exp-1', force=True)
    run()
    check_result()
