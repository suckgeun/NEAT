import gym
from players.manager import Manager
import os


def run_episode(env, manager, nn, i, n_epi=1, render=False):
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
                fitness_epi += 300
                fitness_sum += fitness_epi
                break

    nn.fitness = fitness_sum / float(n_epi)
    if nn.is_champ:
        print("champion agent: {0}".format(nn.fitness_previous))
    print("agent: {0}, raw score: {1}".format(i, nn.fitness - 300))


def run():
    env = gym.make('MountainCar-v0')
    n_input = 2
    n_output = env.action_space.n
    n_nn = 150
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=2, c2=2, c3=1, bias=1, drop_rate=0.8)
    manager.initialize()

    dir_path = os.path.join(os.getcwd(), "results")
    file_path = os.path.join(dir_path, "mountain_fitness_history.txt")
    for i_episode in range(30):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode(env, manager, nn, i, n_epi=3)
        manager.remember_best_nn()
        with open(file_path, "a") as f:
            f.write(str(manager.nn_best.fitness - 300))
            f.write("\n")
        manager.create_next_generation()

    manager.write_best_nn("mountain_result.txt")


def check_result():
    env = gym.make('MountainCar-v0')
    n_input = 2
    n_output = env.action_space.n
    n_nn = 1
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=2, c2=2, c3=1, bias=1, drop_rate=0.8)
    nn = manager.recreate_best_nn("mountain_result.txt")

    for i_episode in range(100):
        print("running episode: {0}".format(i_episode))
        run_episode(env, manager, nn, -1, n_epi=3, render=True)


if __name__ == "__main__":
    run()
    check_result()
