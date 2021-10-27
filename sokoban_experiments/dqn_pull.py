import gym
import gym_sokoban

from stable_baselines.deepq.policies import LnCnnPolicy 
from stable_baselines.common.vec_env import DummyVecEnv 
from stable_baselines import DQN

env = gym.make('PushAndPull-Sokoban-v0')

model = DQN(LnCnnPolicy, env, 
        tensorboard_log=".dqn/", 
        double_q=True,
        exploration_fraction=0.3,
        prioritized_replay=True,
        prioritized_replay_alpha=0.99,
        learning_starts=10_000,
        verbose=1
        )

model.learn(total_timesteps=10_000_000)
model.save(".dqn/dqn_boxoban")

del model # remove to demonstrate saving and loading

model = DQN.load(".dqn/dqn_boxoban")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
