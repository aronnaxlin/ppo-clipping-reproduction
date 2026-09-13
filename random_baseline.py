"""Average return of a uniform random policy per env: the 0-point of the paper's normalized score."""

import json

import gymnasium as gym
import numpy as np

from run_all import ENVS

EPISODES = 100


def main():
    result = {}
    for env_id in ENVS:
        env = gym.make(env_id)
        env.action_space.seed(0)
        returns = []
        obs, _ = env.reset(seed=0)
        ep_ret = 0.0
        while len(returns) < EPISODES:
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            ep_ret += r
            if terminated or truncated:
                returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
        result[env_id] = float(np.mean(returns))
        print(f"{env_id:28s} {result[env_id]:10.2f}")
    with open("random_baseline.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
