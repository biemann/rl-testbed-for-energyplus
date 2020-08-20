
from mpi4py import MPI
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_arg_parser, \
    energyplus_logbase_dir

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# from baselines.run_mpi import run_mpi
# from baselines.ppo1.mlp_policy import MlpPolicy
# from baselines import logger

import os
import datetime
import baselines.common.tf_util as util
import gym_energyplus                        # We need this!

#
# def policy_fn(name, ob_space, ac_space):
#     return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)


def train(env_id, num_timesteps, seed):
    sess = util.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    # Create a new base directory like //home/marco/Reinforcement_Learning/Logs/openai-2018-05-21-12-27

    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        exit()
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        exit()

    # MPI is to parallelize training
    # Logs the training in a file log.txt in the given directory

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print('train: init logger with dir={}'.format(log_dir))  # XXX
        logger.configure(log_dir)
    else:
        logger.configure(format_strings=[])
        logger.set_level(logger.DISABLED)

    # Make Gym environment:

    env = make_energyplus_env(env_id, workerseed)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)


    # Apply TRPO algorithm from OpenAI baselines:

    # Policy_fn : Network given by MlpPolicy
    # num_timesteps : given by parsing (num episodes = num_timesteps/timesteps_per_batch)
    # max_kl: Kullback-Leibner threshold
    # cg_iters: number of iterations for conjugate gradient calculation
    # cg_damping: Compute gradient dampening factor
    # lambda: GAE factor
    # vf_iter: Number of value Function iterations
    # vf_stepsize: Value Function stepsize
    #
    # run_mpi.learn(env, policy_fn,
    #                max_timesteps=num_timesteps, timesteps_per_batch=16 * 1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                 gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # model = TRPO(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=num_timesteps)

    model_ppo = PPO('MlpPolicy', env=env, verbose=1)
    model_td3 = TD3(MlpPolicy, env, verbose=1)
    model_sac = SAC('MlpPolicy', env, verbose=2, learning_starts=1)

    model_sac.learn(total_timesteps=num_timesteps, log_interval=1)

    env.close()


def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
