import gym
import gym_sokoban
 
from stable_baselines.deepq.policies import LnCnnPolicy 
from stable_baselines.common.vec_env import DummyVecEnv 
from stable_baselines import DQN
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList
 
path = F".dqn/" 
STEPS_PER_SAVE = 10_000
 
env = gym.make('Sokoban-small-v0')
model = DQN(LnCnnPolicy, env, 
        tensorboard_log=path+"GRAPH/", 
        double_q=True,
        prioritized_replay=True,
        prioritized_replay_alpha=0.99,
        learning_starts=1000,
        verbose=1
        )

for i in range(1000):
  print("=" * 50, " ENVIRONMENT #", i, " ", "=" * 50)
  env.reset(regenerate=True)
  model.set_env(env)
  callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=9, verbose=1)
  eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
  callback_checkpoint = CheckpointCallback(save_freq=STEPS_PER_SAVE, save_path=path+"MODEL/", name_prefix="dqn_pr", verbose=0)
  callback_list = CallbackList([eval_callback, callback_checkpoint])
  model.learn(total_timesteps=STEPS_PER_SAVE, callback=callback_list)
  model.save(path+"MODEL/final_model_for_env_"+str(i))
