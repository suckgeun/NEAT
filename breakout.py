import gym
import numpy as np
from players.manager import Manager


def get_refined_observation_breakout(observation):
    observation = observation[100:-20, 10:-10]
    hey = observation.sum(axis=2)

    refined = np.array([])
    div_210 = 10
    div_160 = 10
    len_y = int(hey.shape[0]/div_210)
    len_x = int(hey.shape[1]/div_160)
    for y in range(len_y):
        for x in range(len_x):
            refined = np.append(refined, np.array(hey[div_210*y:div_210*(y+1), div_160*x:div_160*(x+1)].sum()))

    refined = (refined > 0).astype(float)
    # refined[0: 1*len_x][refined[0: 1*len_x] > 0] = 20
    # refined[1*len_x: 6*len_x][refined[1*len_x: 6*len_x] > 0] = 1
    refined[6*len_x: 14*len_x][refined[6*len_x: 14*len_x] > 0] = 20
    refined[14*len_x: 15*len_x][refined[14*len_x: 15*len_x] > 0] = 10

    # dx = int(hey.shape[1]/div_160)
    # for i in range(int(hey.shape[0]/div_210)):
    #     print(refined[dx*i: dx*(i+1)])
    #
    # print(refined.shape)

    return refined


def run_episode_breakout(env, manager, nn, i, count):
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

            if nn.fitness > 1000:
                name = "result" + str(count) + ".txt"
                f = open(name, "w")
                f.write(str(nn.connect_genes))
                f.write(str(nn.fitness))
                f.close()
            break


def run_breakout():
    env = gym.make('Breakout-v0')
    n_input = 9 * 14
    n_output = env.action_space.n
    n_nn = 150
    print("number of input: {0}, number of output:{1}".format(n_input, n_output))
    manager = Manager(n_nn, n_input, n_output, c1=400, c2=400, c3=250, bias=1)
    manager.initialize()

    count = 0
    for i_episode in range(200):
        print("running episode: {0}".format(i_episode))
        for i, nn in enumerate(manager.workplace.nns):
            run_episode_breakout(env, manager, nn, i, count)
            count += 1
        manager.create_next_generation()


def run():
    run_breakout()

if __name__ == "__main__":
    run()
