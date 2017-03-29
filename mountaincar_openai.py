import gym
from players.manager import Manager
import os
from gym import wrappers


def run_episode(env, manager, nn, i, n_epi=1, render=False):
    """
    runs each episode

    :param env: OpenAI environment
    :param manager: NEAT manager
    :param nn: neural network to run the episode
    :param i: neural network index
    :param n_epi: number of times to run episode
    :param render: if True, renders the game play
    """
    observation = env.reset()
    nn.fitness = 0
    fitness_sum = 0

    for epi in range(n_epi):
        fitness_epi = 0
        for step in range(300):

            if render:
                env.render()

            action = manager.get_action(observation, nn)
            observation, reward, done, info = env.step(action)
            fitness_epi += reward

            if done:
                fitness_epi += 200
                fitness_sum += fitness_epi
                observation = env.reset()
                break

    nn.fitness = fitness_sum / float(n_epi)
    if nn.is_champ:
        print("champion agent: {0}".format(nn.fitness_previous - 200))
    print("agent: {0}, raw score: {1}".format(i, nn.fitness - 200))


def run():
    """
    Runs NEAT training
    """
    n_input = 2
    n_output = env.action_space.n
    n_nn = 150
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=2, c2=2, c3=1, bias=1, drop_rate=0.8, weight_max=10,
                      weight_min=-10, weight_mutate_rate=0.01, pm_weight_random=0.2)
    manager.initialize()

    dir_path = os.path.join(os.getcwd(), "results")
    file_path = os.path.join(dir_path, "mountain_fitness_history_openai.txt")

    # run 30 generations
    for i_episode in range(30):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode(env, manager, nn, i, n_epi=3)

            if nn.fitness >= -108 + 200:
                print("### checking best nn ###")

                run_episode(env, manager, nn, i, n_epi=20)

                if nn.fitness >= -108 + 200:

                    # record best fitness
                    manager.remember_best_nn(nn)

                    with open(file_path, "a") as f:
                        f.write(str(manager.nn_best.fitness - 200))
                        f.write("\n")

                    # record best neural network
                    manager.write_best_nn("mountain_result_openai.txt")
                    return

        manager.create_next_generation()


def check_result():
    """
    Validates the best performed neural network
    """
    n_input = 2
    n_output = env.action_space.n
    n_nn = 1
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=2, c2=2, c3=1, bias=1, drop_rate=0.8, weight_max=10,
                      weight_min=-10, weight_mutate_rate=0.1, pm_weight_random=0.3)
    manager.initialize()
    nn = manager.recreate_best_nn("mountain_result_openai.txt")

    # runs 100 times and see the results
    for i_episode in range(100):
        print("running episode: {0}".format(i_episode))
        run_episode(env, manager, nn, -1, n_epi=3, render=True)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = wrappers.Monitor(env, 'tmp/mountaincar-exp-1', force=True)
    run()
    check_result()
