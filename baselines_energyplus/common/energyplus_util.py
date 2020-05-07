"""
Helpers for script run_energyplus.py.
"""

import os
import gym
from baselines import logger
from baselines_energyplus.bench import Monitor
import glob


# Makes Energy Plus Environment, following OpenAI gym

def make_energyplus_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for EnergyEnv
    """

    # Makes an instance of an OpenAI gym environment, called EnergyPlus-v0, using the information in energyplus_env.py
    env = gym.make(env_id)

    # We make an instance of the Monitor class, based on the OpenAI gym environment
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for run_energypl.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(100))
    parser.add_argument('--save-interval', type=int, default=int(0))
    parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    return parser


# Reads the logs from the environment variables in order to show the plot:

def energyplus_locate_log_dir(index=0):
    pat = energyplus_logbase_dir() + '/openai-????-??-??-??-??*/progress.csv'
    files = [(f, os.path.getmtime(f)) for f in glob.glob(pat)]
    newest = sorted(files, key=lambda fil: fil[1])[-(1 + index)][0]
    directory = os.path.dirname(newest)
    print('energyplus_locate_log_dir: {}'.format(directory))
    return directory


def energyplus_logbase_dir():
    logbase_dir = os.getenv('ENERGYPLUS_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir
