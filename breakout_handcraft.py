import gym
import numpy as np


def get_refined_observation_breakout(observation):
    whole_area = observation[93:-18, 8:-8]
    ball_area = whole_area[:96, :]
    bar_area = whole_area[96:, :]

    ball_area_num = ball_area.sum(axis=2)
    bar_area_num = bar_area.sum(axis=2)

    ball_x = 0
    bar_x = 0

    for i, row in enumerate(ball_area_num):
        index = np.where(row != 0)[0]
        if len(index) != 0:
            ball_x = (index[0] - 72.0)
            break

    for i, row in enumerate(bar_area_num):
        index = np.where(row != 0)[0]
        if len(index) != 0:
            bar_x = (index[0]-72.0) + 8
            break

    refined = np.array([ball_x, bar_x])

    return refined


def run_episode_breakout(env):
    observation = env.reset()

    for step in range(40000):
        print("step: {0}".format(step))
        env.render()
        refined_obsv = get_refined_observation_breakout(observation)
        # print(env.action_space)

        ball_x = refined_obsv[0]
        bar_x = refined_obsv[1]

        diff = (ball_x - bar_x) / float(ball_x)
        print(diff)

        if step == 0:
            action = 1
        elif diff > 0.1:
            action = 2
        elif -0.1 <= diff <= 0.1:
            action = 0
        else:
            action = 3
        print("ball: {0}, bar: {1}, action: {2}".format(ball_x, bar_x, action))

        observation, reward, done, info = env.step(action)

        if done:
            print(reward)
            break


def run_breakout():
    env = gym.make('Breakout-v0')
    print("hand crafted agent")

    for i_episode in range(200):
        print("running episode: {0}".format(i_episode))
        run_episode_breakout(env)


def run():
    run_breakout()

if __name__ == "__main__":
    run()
