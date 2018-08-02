import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic
from tensorboardX import SummaryWriter


def test(rank, args, shared_model, counter):
    with torch.no_grad():
        my_writer = SummaryWriter(log_dir='log')
        t0 = time.time()
        torch.manual_seed(args.seed + rank)

        env = create_atari_env(args.env_name)
        env.seed(args.seed + rank)

        model = ActorCritic(env.observation_space.shape[0], env.action_space)
        model.eval()

        state = env.reset()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()

        # a quick hack to prevent the agent from stucking
        actions = deque(maxlen=100)
        episode_length = 0
        while True:
            episode_length += 1
            # Sync with the shared model
            if done:
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit, dim=1)
            action = prob.max(1, keepdim=True)[1].data.numpy()

            state, reward, done, _ = env.step(action[0, 0])
            done = done or episode_length >= args.max_episode_length
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                my_writer.add_scalar('episode_reward', reward_sum, counter.value)
                my_writer.add_scalar('episode_length', episode_length, counter.value)
                my_writer.add_scalar('FPS', counter.value / (time.time() - start_time), counter.value)
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length))
                reward_sum = 0
                episode_length = 0
                actions.clear()
                state = env.reset()
                time.sleep(4 * 60)
                if (time.time() - t0) > (15 * 60):
                    torch.save(model.state_dict(), os.path.join('models', 'epoch_{}.pth').format(str(int(time.time()))))
                    t0 = time.time()

            state = torch.from_numpy(state)
