import gym
import numpy as np
from players.manager import Manager


def get_refined_observation_breakout(observation):
    whole_area = observation[93:-18, 8:-8]
    ball_area = whole_area[:96, :]
    bar_area = whole_area[96:, :]

    ball_area_num = ball_area.sum(axis=2)
    bar_area_num = bar_area.sum(axis=2)

    ball_x = 0
    ball_y = 0
    bar_x = 0
    bar_y = 0

    for i, row in enumerate(ball_area_num):
        index = np.where(row != 0)[0]
        if len(index) != 0:
            ball_x = (index[0] - 72.0)/144.0
            ball_y = i/96.0
            break

    for i, row in enumerate(bar_area_num):
        index = np.where(row != 0)[0]
        if len(index) != 0:
            bar_x = (index[0]-72.0)/144.0
            bar_y = i+96
            break

    # refined = np.array([ball_x, ball_y, bar_x, bar_y])
    refined = np.array([ball_x, ball_y, bar_x])

    return refined


def run_episode_breakout(env, manager, nn, i):
    observation = env.reset()
    nn.fitness = 0
    reward_prev = 0
    action_prev = 0
    count = 0
    for step in range(40000):

        # env.render()
        refined_obsv = get_refined_observation_breakout(observation)
        # print(env.action_space)

        action = manager.get_action(refined_obsv, nn)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        nn.fitness += reward

        if reward_prev == reward and action_prev == action:
            count += 1
            if count > 300:
                if nn.is_champ:
                    print("champion agent: {0}".format(nn.fitness_previous))
                print("agent: {0}, raw score: {1}, steps: {2}".format(i, nn.fitness, step))
                break

        else:
            count = 0

        reward_prev = reward
        action_prev = action
        if done:

            if nn.is_champ:
                print("champion agent: {0}".format(nn.fitness_previous))
            print("agent: {0}, raw score: {1}, steps: {2}".format(i, nn.fitness, step))
            break


def run_breakout():
    env = gym.make('Breakout-v0')
    n_input = 3
    n_output = env.action_space.n
    n_nn = 300
    nn_best = None
    score_best = 0
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=3, c2=3, c3=2, bias=1)
    manager.initialize()

    for i_episode in range(200):
        is_new_champ = False
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode_breakout(env, manager, nn, i)
            if nn_best is None:
                nn_best = nn
            elif score_best < nn.fitness:
                print("prev_best: {0}, new_best: {1}".format(nn_best.fitness, nn.fitness))
                nn_best = nn
                score_best = nn.fitness
                is_new_champ = True

        name = "result" + str(i_episode) + ".txt"

        if is_new_champ:
            f = open(name, "w")
            f.write(str(nn_best.connect_genes[:, :5]))
            f.write("\n")
            f.write(str(nn_best.fitness))
            f.close()

        manager.create_next_generation()


def run():
    run_breakout()

if __name__ == "__main__":
    run()
