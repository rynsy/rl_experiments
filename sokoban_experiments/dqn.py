import gym
import gym_sokoban
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.deepq.policies import CnnPolicy 
from stable_baselines.common.vec_env import DummyVecEnv 
from stable_baselines import DQN
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
 
path = F".dqn/" 
model_path = path + "MODEL/"
graph_path = path + "GRAPH/"
pre_trained_path = model_path + "pretrained_dqn"
MAX_STEPS = 10_000

env = gym.make('Sokoban-small-v0')

def new_model(env):
    model = DQN(CnnPolicy, env,
#                tensorboard_log=graph_path,
                double_q=True,
                prioritized_replay=True,
                prioritized_replay_alpha=0.99,
                exploration_fraction=0.5,
                learning_starts=MAX_STEPS * 0.5,
                verbose=1
            )
    return model

def pre_trained_model(env):
    model = DQN(CnnPolicy, env,
#                tensorboard_log=graph_path,
                double_q=True,
                prioritized_replay=True,
                prioritized_replay_alpha=0.99,
                learning_starts=MAX_STEPS * 0.5,
                verbose=1
            )
    model.load(pre_trained_path) 
    return model


def test_model(env, model, regen=False):
  test_rewards = []
  for i in range(10):
    test_reward = []
    obs = env.reset(regenerate=regen)
    while True:
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)
      env.render()
      test_reward.append(rewards)
      if done:
        break
    test_reward = np.sum(test_reward)
    test_rewards.append(test_reward)
    print("Test {}: {}".format(i,test_reward))
  if regen:
    print("Average reward for random-board tests: {}".format(np.average(test_rewards)))
  else:
    print("Average reward for same-board tests: {}".format(np.average(test_rewards)))
  return np.average(test_rewards)

def single_boards(env, model):
    for i in range(1000):
      print("=" * 50, " ENVIRONMENT #", i, " ", "=" * 50)
      board_learned = False
      while not board_learned:
        env.reset(regenerate=board_learned)
        model.set_env(env)
        model.learn(total_timesteps=MAX_STEPS)
        reward = test_model(env, model)
        board_learned = reward > 9
        if board_learned:
          test_model(env, model, regen=board_learned)
          model.save(model_path+"final_model_for_env_"+str(i))
          model.save(pre_trained_path)
          print("=" * 50, " ENVIRONMENT #", i, " HAS BEEN LEARNED ", "=" * 50)

if __name__ == "__main__":
    env = gym.make('Sokoban-small-v0')
    model = new_model(env)
    single_boards(env, model)
