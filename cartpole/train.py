"""Training in NumPy: REINFORCE with a baseline, actor-critic, and a cross-entropy-method search.

Policy gradient in one paragraph. The policy pi(a|s) is a softmax over two logits. An episode
is a sequence (s_t, a_t, r_t). The objective is J = E[sum_t r_t], and its gradient is

    grad J = E[ sum_t grad log pi(a_t|s_t) * A_t ],

where A_t is an advantage: how much better the action was than what the policy usually gets
from that state. Subtracting any baseline b(s_t) from the return leaves the gradient unbiased
but lowers its variance, so the advantage is the return minus a baseline. Gradient ascent on
J is implemented as descent on the loss L = -(1/T) sum_t A_t log pi(a_t|s_t). For a softmax,
d log pi(a|z) / dz_k = 1[k = a] - p_k, so the gradient with respect to the logits is simply
-(A_t / T) * (onehot(a_t) - p_t), which is what the code below feeds into backprop.

REINFORCE uses the Monte Carlo return G_t (discounted sum of the rewards that follow) as the
signal, with the batch mean as the baseline and the batch standard deviation to normalise.
Actor-critic replaces the batch mean with a learned critic V(s): the advantage is G_t - V(s_t),
and the critic regresses onto the same returns. The shipped variant uses Monte Carlo returns as
the critic's target. The textbook alternative, a one-step TD target r_t + gamma V(s_{t+1}) with
advantage delta_t = r_t + gamma V(s_{t+1}) - V(s_t), is kept as ACTOR_CRITIC_TARGET = "td". In
our runs it did not learn reliably (see the notes in results/cartpole_learning.md).

The cross-entropy method (CEM) skips gradients altogether. It samples linear policies from a
Gaussian, keeps the best quarter, and refits the Gaussian to them. It is a useful comparison
because the policy is tiny and the search space is only five numbers.

Every run is seeded: the same seed gives the same weights, episodes and curve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agents import LinearAgent
from .env import CartPole, run_episode
from .nets import MLP, Adam, normalise, save_npz, softmax

GAMMA = 0.99          # discount factor
BATCH = 8             # episodes per REINFORCE gradient update
CRITIC_BATCH = 8      # episodes per actor-critic update
ACTOR_CRITIC_TARGET = "mc"  # critic regresses onto the Monte Carlo return ("mc") or the TD(0) target ("td")
HIDDEN = 16           # tanh units in each MLP
LR = 1e-2             # Adam step size for the policy
CRITIC_LR = 1e-2      # Adam step size for the critic (its output is multiplied by VALUE_SCALE)
VALUE_SCALE = 50.0    # the critic predicts V / VALUE_SCALE, so its outputs stay near 1 (V can reach ~100)
CEM_POP = 40          # linear policies sampled per generation
CEM_ELITE = 10        # the best quarter refits the sampling distribution
CEM_NOISE = 0.01      # floor on the sampling std, so the search never collapses to a point too early
ALGOS = ("reinforce", "actor_critic", "cem")

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Episode:
    """One rollout, stored as arrays so a whole batch can be processed with matrix ops."""
    obs: np.ndarray      # (T, 4) normalised observations the agent saw
    act: np.ndarray      # (T,) actions taken
    rew: np.ndarray      # (T,) rewards, all 1.0 for CartPole
    nxt: np.ndarray      # (T, 4) normalised observation after each step
    terminated: bool     # True if the pole fell or the cart left the track; False if truncated at 500

    @property
    def steps(self) -> int:
        return len(self.act)


def rollout(policy: MLP, rng: np.random.Generator, env_seed: int) -> Episode:
    """Sample one episode from the stochastic policy. Training uses samples, not the argmax."""
    env = CartPole(env_seed)
    o = env.reset(env_seed)
    obs, act, rew, nxt = [], [], [], []
    while not env.done:
        x = normalise(o)
        _, Z = policy.forward(x[None, :])
        p_left = softmax(Z)[0, 0]
        a = 0 if rng.random() < p_left else 1
        o, r, _, _ = env.step(a)
        obs.append(x)
        act.append(a)
        rew.append(r)
        nxt.append(normalise(o))
    return Episode(np.array(obs), np.array(act), np.array(rew), np.array(nxt), env.terminated)


def discounted_returns(rew: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """G_t = r_t + gamma * G_{t+1}, computed backwards. Truncated episodes simply end at the cap."""
    G = np.zeros_like(rew)
    g = 0.0
    for t in range(len(rew) - 1, -1, -1):
        g = rew[t] + gamma * g
        G[t] = g
    return G


def policy_gradient_step(policy: MLP, opt: Adam, X, A, adv) -> None:
    """One gradient-descent step on L = -(1/T) sum_t adv_t log pi(a_t|x_t), given advantages.

    dL/dz = -(adv / T) * (onehot(a) - softmax(z)); backprop turns that into the parameter gradients.
    """
    H, Z = policy.forward(X)
    P = softmax(Z)
    onehot = np.eye(2)[A]
    dZ = -(adv[:, None] / len(A)) * (onehot - P)
    opt.step(policy.backward(X, H, dZ))


def reinforce_update(policy: MLP, opt: Adam, batch: list[Episode]) -> None:
    """Baseline = mean return over the batch; the advantage is also divided by the batch std."""
    X = np.concatenate([e.obs for e in batch])
    A = np.concatenate([e.act for e in batch])
    G = np.concatenate([discounted_returns(e.rew) for e in batch])
    adv = (G - G.mean()) / (G.std() + 1e-8)
    policy_gradient_step(policy, opt, X, A, adv)


def actor_critic_update(policy: MLP, opt_pi: Adam, critic: MLP, opt_v: Adam, batch: list[Episode]) -> None:
    """Advantage = critic error. The critic's target is the Monte Carlo return by default, or the TD(0)
    target (bootstrapping from V(s') and stopping at a true failure, not at the 500-step cap) when
    ACTOR_CRITIC_TARGET is "td"."""
    X = np.concatenate([e.obs for e in batch])
    Xn = np.concatenate([e.nxt for e in batch])
    A = np.concatenate([e.act for e in batch])
    R = np.concatenate([e.rew for e in batch])
    # done = 1 only on the last step of an episode that terminated. The value of a fallen pole is 0.
    done = np.concatenate([np.r_[np.zeros(e.steps - 1), float(e.terminated)] for e in batch])
    T = len(X)

    Hv, Zv = critic.forward(X)
    V = VALUE_SCALE * Zv[:, 0]
    if ACTOR_CRITIC_TARGET == "mc":
        y = np.concatenate([discounted_returns(e.rew) for e in batch])
    else:
        _, Zn = critic.forward(Xn)
        Vn = VALUE_SCALE * Zn[:, 0]
        y = R + GAMMA * Vn * (1.0 - done)   # TD target; treated as a constant when differentiating

    # Critic: minimise 0.5 * mean (V - y)^2. dL/dZ = VALUE_SCALE * (V - y) / T.
    dZv = (VALUE_SCALE * (V - y) / T)[:, None]
    opt_v.step(critic.backward(X, Hv, dZv))

    # Actor: the critic's error is the advantage. It is computed before the critic moves, from the same values.
    # Dividing by VALUE_SCALE brings it to the size of a typical return-minus-baseline gap, roughly unit scale.
    delta = y - V
    policy_gradient_step(policy, opt_pi, X, A, delta / VALUE_SCALE)


def _init_policy(rng: np.random.Generator) -> MLP:
    # out_scale 0.1: the starting policy is close to 50/50.
    return MLP.init(rng, 4, HIDDEN, 2, out_scale=0.1)


def train_policy_gradient(algo: str, seed: int, episodes: int, log=None):
    """Train REINFORCE or actor-critic from scratch. Returns (weights, per-episode step counts)."""
    rng = np.random.default_rng(seed)
    policy = _init_policy(rng)
    opt_pi = Adam(policy.params, LR)
    critic = opt_v = None
    if algo == "actor_critic":
        critic = MLP.init(rng, 4, HIDDEN, 1, out_scale=1.0)
        opt_v = Adam(critic.params, CRITIC_LR)
    returns: list[int] = []
    batch_size = BATCH if algo == "reinforce" else CRITIC_BATCH
    while len(returns) < episodes:
        batch = []
        for _ in range(batch_size):
            env_seed = int(rng.integers(2**31))  # each episode gets its own start state
            batch.append(rollout(policy, rng, env_seed))
        returns.extend(e.steps for e in batch)
        if algo == "reinforce":
            reinforce_update(policy, opt_pi, batch)
        else:
            actor_critic_update(policy, opt_pi, critic, opt_v, batch)
        if log and len(returns) % 200 < BATCH:
            log(f"  {algo} seed {seed}: {len(returns)} episodes, last 50 mean {np.mean(returns[-50:]):.1f}")
    weights = dict(policy.params)
    if critic is not None:
        weights.update({f"v_{k}": v for k, v in critic.params.items()})
        weights["value_scale"] = np.array([VALUE_SCALE])
    return weights, returns[:episodes]


def train_cem(seed: int, episodes: int, log=None):
    """Cross-entropy method over the 5 parameters of a linear policy (4 weights and a bias)."""
    rng = np.random.default_rng(seed)
    dim = 5
    mu = np.zeros(dim)
    sigma = np.ones(dim)
    returns: list[int] = []
    while len(returns) < episodes:
        thetas = mu + sigma * rng.standard_normal((CEM_POP, dim))
        # All candidates in a generation face the same start states, so they are compared fairly.
        env_seed = int(rng.integers(2**31))
        scores = [run_episode(LinearAgent(th), env_seed) for th in thetas]
        returns.extend(scores)
        elite = thetas[np.argsort(scores)[-CEM_ELITE:]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + CEM_NOISE
        if log and len(returns) % 400 < CEM_POP:
            log(f"  cem seed {seed}: {len(returns)} episodes, generation best {max(scores)}")
    return {"theta": mu}, returns[:episodes]


def train(algo: str, seed: int, episodes: int, log=None):
    if algo == "cem":
        return train_cem(seed, episodes, log)
    if algo in ("reinforce", "actor_critic"):
        return train_policy_gradient(algo, seed, episodes, log)
    raise ValueError(f"unknown algorithm {algo!r}; choose from {', '.join(ALGOS)}")


def learning_summary(returns: list[int]) -> dict:
    """Numbers that describe one learning curve: the first and last 50 episodes, and when it first reached 475."""
    r = np.asarray(returns, dtype=float)
    out = {"episodes": len(r), "first50": float(r[:50].mean()), "last50": float(r[-50:].mean())}
    window = 100
    if len(r) >= window:
        trailing = np.convolve(r, np.ones(window) / window, mode="valid")
        hit = np.flatnonzero(trailing >= 475)
        out["episodes_to_475_trailing100"] = int(hit[0] + window) if len(hit) else None
    return out


def train_and_save(algo: str, seeds: list[int], episodes: int, data_dir: Path = DATA_DIR, log=print) -> dict:
    """Train each seed, keep the weights of the seed whose final 50 episodes did best, and merge curves.

    Ties go to the seed listed first: `max` keeps the first of equal scores. Several seeds often reach the
    500-step cap, so their last-50 means tie at 500 and the first listed seed is shipped.
    Selection uses training episodes only, so the benchmark's evaluation seeds stay unseen.
    The curves file keeps every seed, which is what the learning-curve chart shows.
    """
    curves = {}
    runs = {}
    for seed in seeds:
        log(f"training {algo} seed {seed} for {episodes} episodes")
        weights, returns = train(algo, seed, episodes, log)
        runs[seed] = weights
        curves[seed] = returns
    best = max(seeds, key=lambda s: np.mean(curves[s][-50:]))
    data_dir.mkdir(parents=True, exist_ok=True)
    save_npz(data_dir / f"{algo}.npz", runs[best])

    path = data_dir / "curves.json"
    doc = json.loads(path.read_text()) if path.exists() else {"algorithms": {}}
    doc["algorithms"][algo] = {
        "best_seed": int(best),
        "seeds": {str(s): [int(x) for x in curves[s]] for s in seeds},
    }
    doc["gamma"] = GAMMA
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return {
        "best_seed": best,
        "summaries": {s: learning_summary(curves[s]) for s in seeds},
    }
