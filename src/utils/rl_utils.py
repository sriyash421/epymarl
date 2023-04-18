import torch as th
import random

try:
    import gfootball.env as football_env
except:
    pass

from gym import spaces
import numpy as np
from gym.envs.registration import registry, register, make, spec

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

class Spec(object):
    max_episode_steps=None

class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, *args, **kwargs):
        self.num_agents = kwargs['num_agents']
        self.scenario_name = kwargs['scenario_name']

        # make env
        # if not (args.use_render and args.save_videos):
        self.env = football_env.create_environment(
            env_name=kwargs['scenario_name'],
            stacked=False,
            representation=kwargs['representation'],
            rewards=kwargs['rewards'],
            number_of_left_players_agent_controls=kwargs['num_agents'],
            number_of_right_players_agent_controls=0,
            channel_dimensions=(96, 72),
            render=False
        )
        # else:
        #     # render env and save videos
        #     self.env = football_env.create_environment(
        #         env_name=args.scenario_name,
        #         stacked=False,
        #         representation=args.representation,
        #         rewards=args.rewards,
        #         number_of_left_players_agent_controls=args.num_agents,
        #         number_of_right_players_agent_controls=0,
        #         channel_dimensions=(96, 72),
        #         # video related params
        #         write_full_episode_dumps=True,
        #         render=True,
        #         write_video=True,
        #         dump_frequency=1,
        #         logdir=args.video_dir
        #     )

        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = False
        self.zero_feature = False
        self.share_reward = True
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        self.unwrapped =  self.env.unwrapped
        self.spec = Spec()
        self.n_agents = self.num_agents
        # print(self.spec)

        if self.num_agents == 1:
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
        else:
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))
                self.share_observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))


    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [global_reward] * self.num_agents

        # done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info

register(
  id="Football-v1",
  entry_point=FootballEnv,
  max_episode_steps=200,
  kwargs=dict(
        scenario_name="academy_3_vs_1_with_keeper",
        num_agents=3,
        episode_length =200,
        stacked=False,
        representation="simple115v2",
        rewards="scoring,checkpoints",
        logdir='/tmp/football',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        number_of_players_agent_controls=6
    ),
)
