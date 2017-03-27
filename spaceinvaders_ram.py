import gym
from players.manager import Manager
import os


def run_episode(env, manager, nn, i, n_epi=1, render=False, random=False):
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
    env = gym.make('SpaceInvaders-ram-v0')
    n_input = 128
    n_output = env.action_space.n
    n_nn = 150
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=300, c2=300, c3=150, bias=1, drop_rate=0.8,
                      weight_max=3, weight_min=-3, weight_mutate_rate=0.1)
    manager.initialize()

    dir_path = os.path.join(os.getcwd(), "results")
    file_path = os.path.join(dir_path, "invaders_fitness_history.txt")
    for i_episode in range(40):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode(env, manager, nn, i, n_epi=3)
        manager.remember_best_nn()
        with open(file_path, "a") as f:
            f.write(str(manager.nn_best.fitness))
            f.write("\n")
        manager.create_next_generation()

    manager.write_best_nn("invaders_result.txt")


def check_result():
    env = gym.make('SpaceInvaders-ram-v0')
    n_input = 128
    n_output = env.action_space.n
    n_nn = 1
    n_trials = 100
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=300, c2=300, c3=100, bias=1, drop_rate=0.8)
    nn = manager.recreate_best_nn("invaders_result.txt")

    fitness_sum = 0
    for i_episode in range(n_trials):
        print("running episode: {0}".format(i_episode))
        fitness = run_episode(env, manager, nn, -1, n_epi=1, render=False, random=True)
        fitness_sum += fitness

    print(fitness_sum / float(n_trials))

if __name__ == "__main__":
    run()
    check_result()
