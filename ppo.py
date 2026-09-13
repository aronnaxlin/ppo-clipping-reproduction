"""PPO (clipped surrogate) for MuJoCo, single environment.

Reproduces the clipping ablation of Schulman et al. 2017, "Proximal Policy
Optimization Algorithms" (arXiv:1707.06347), Table 1.

Algorithm details follow CleanRL's ppo_continuous_action.py
(https://github.com/vwxyzjn/cleanrl), which matches openai/baselines ppo2.
Deviations from CleanRL, all deliberate:
  * one plain env instead of SyncVectorEnv (gymnasium 1.x changed vector
    autoreset semantics; with num_envs=1 a plain env is simpler and exact);
  * value-loss clipping is OFF, so the clip coefficient only touches the policy
    objective -- the quantity the paper ablates;
  * `--clip none` gives the paper's "no clipping or penalty" objective r_t * A_t;
  * extra per-update diagnostics (ratio range, KL, clip fraction).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

torch.set_num_threads(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--clip", default="0.2", help="clip coefficient epsilon, or 'none'")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--out-dir", default="runs")
    # Paper Table 3
    p.add_argument("--num-steps", type=int, default=2048)  # horizon T
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=10)  # K
    p.add_argument("--minibatch-size", type=int, default=64)  # M
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    # Not in the paper; openai/baselines ppo2 / CleanRL defaults
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--norm-adv", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)  # paper: no entropy bonus
    return p.parse_args()


def make_env(env_id, gamma):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # raw returns, before any reward scaling
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    return env


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Separate policy and value MLPs, 2x64 tanh (paper Sec. 6.1)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))  # state-independent

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        mean = self.actor_mean(x)
        dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(x)


def main():
    args = parse_args()
    clip = None if args.clip.lower() == "none" else float(args.clip)
    variant = "noclip" if clip is None else f"clip{clip}"
    run_dir = Path(args.out_dir) / args.env_id / variant / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.env_id, args.gamma)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    T = args.num_steps
    num_updates = args.total_timesteps // T
    obs_buf = torch.zeros((T, obs_dim))
    act_buf = torch.zeros((T, act_dim))
    logp_buf = torch.zeros(T)
    rew_buf = torch.zeros(T)
    done_buf = torch.zeros(T)
    val_buf = torch.zeros(T)

    ep_file = open(run_dir / "episodes.csv", "w", newline="")
    ep_log = csv.writer(ep_file)
    ep_log.writerow(["global_step", "return", "length"])
    up_file = open(run_dir / "updates.csv", "w", newline="")
    up_log = csv.writer(up_file)
    up_log.writerow(["global_step", "lr", "policy_loss", "value_loss", "entropy", "approx_kl",
                     "clipfrac", "ratio_min", "ratio_max", "explained_var", "sps"])

    global_step = 0
    start = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    next_done = 0.0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * args.learning_rate

        # ---- rollout: T steps with pi_old ----
        for t in range(T):
            global_step += 1
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.act(next_obs.unsqueeze(0))
            act_buf[t] = action[0]
            logp_buf[t] = logp[0]
            val_buf[t] = value[0]

            o, r, terminated, truncated, info = env.step(action[0].numpy())
            rew_buf[t] = float(r)
            next_done = float(terminated or truncated)
            if next_done:
                ep_log.writerow([global_step, float(info["episode"]["r"]), int(info["episode"]["l"])])
                o, _ = env.reset()
            next_obs = torch.tensor(o, dtype=torch.float32)

        # ---- truncated GAE (paper Eq. 11-12) ----
        with torch.no_grad():
            next_value = agent.value(next_obs.unsqueeze(0))[0]
            adv = torch.zeros(T)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nonterminal, nextv = 1.0 - next_done, next_value
                else:
                    nonterminal, nextv = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextv * nonterminal - val_buf[t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
            returns = adv + val_buf

        # ---- K epochs of minibatch Adam on the surrogate ----
        inds = np.arange(T)
        clipfracs, kls = [], []
        rmin, rmax = float("inf"), float("-inf")
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, T, args.minibatch_size):
                mb = inds[s:s + args.minibatch_size]
                _, newlogp, entropy, newv = agent.act(obs_buf[mb], act_buf[mb])
                logratio = newlogp - logp_buf[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    kls.append(((ratio - 1) - logratio).mean().item())
                    # for noclip, report the fraction outside the eps=0.2 band for comparison
                    clipfracs.append(((ratio - 1).abs() > (clip or 0.2)).float().mean().item())
                    rmin, rmax = min(rmin, ratio.min().item()), max(rmax, ratio.max().item())

                mb_adv = adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                if clip is None:
                    pg_loss = -(mb_adv * ratio).mean()  # L^CPI, Eq. 6
                else:  # L^CLIP, Eq. 7 (negated for minimisation)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                v_loss = 0.5 * ((newv - returns[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        var_y = returns.var().item()
        ev = float("nan") if var_y == 0 else 1 - (returns - val_buf).var().item() / var_y
        up_log.writerow([global_step, optimizer.param_groups[0]["lr"], pg_loss.item(), v_loss.item(),
                         ent.item(), np.mean(kls), np.mean(clipfracs), rmin, rmax, ev,
                         int(global_step / (time.time() - start))])
        ep_file.flush()
        up_file.flush()

    ep_file.close()
    up_file.close()
    env.close()

    returns_all = np.loadtxt(run_dir / "episodes.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1]
    summary = {
        "env_id": args.env_id, "variant": variant, "seed": args.seed,
        "episodes": int(len(returns_all)),
        "last100_mean_return": float(returns_all[-100:].mean()) if len(returns_all) else float("nan"),
        "wall_time_s": round(time.time() - start, 1),
    }
    (run_dir / "done.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
