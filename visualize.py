import os
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import gym
import imageio

from envs import create_atari_env
from model import ActorCritic


def visualize(env_name, model_path, render=False):
    with torch.no_grad():
        torch.manual_seed(0)

        env = create_atari_env(env_name)
        env_orig = gym.make(env_name)
        env.seed(0)
        env_orig.seed(0)

        model = ActorCritic(env.observation_space.shape[0], env.action_space)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        model.eval()

        states = []
        state, state_orig = env.reset(), env_orig.reset()
        states.append(state_orig)
        state = torch.from_numpy(state)
        reward_sum = 0

        # a quick hack to prevent the agent from stucking
        actions = deque(maxlen=1000)
        episode_length = 0

        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
        while True:
            episode_length += 1

            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1, keepdim=True)[1].data.numpy()

            state, reward, done, _ = env.step(action[0, 0])
            state_orig, _, _, _ = env_orig.step(action[0, 0])
            states.append(state_orig)
            state = torch.from_numpy(state)
            if render:
                env.render()
            done = done or episode_length >= 10000
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                print('stuck in infinite loop')
                done = True

            if done:
                print('episode_length: {}'.format(episode_length))
                print('reward_sum: {}'.format(reward_sum))
                break

            cx = Variable(cx.data)
            hx = Variable(hx.data)

    env.close(), env_orig.close()
    return states

if __name__ == '__main__':
    MODEL_DIR = 'models'
    VIDEO_DIR = 'videos'
    for model_name in sorted(os.listdir(MODEL_DIR)):
        print('play with model {}'.format(model_name))
        result = visualize('BreakoutNoFrameskip-v4', os.path.join(MODEL_DIR, model_name))
        imageio.mimwrite(os.path.join(VIDEO_DIR, model_name.replace('.pth', '.mp4')), result , fps=60)
        print('-' * 40)
