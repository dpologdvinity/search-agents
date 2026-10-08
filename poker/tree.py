"""Explicit game tree for a two-player zero-sum game, plus exact value and exploitability.

Why an explicit tree: CFR needs, at every decision, the regret of each action summed over all
histories in the same information set, weighted by the other player's and the chance reach.
Kuhn (about 30 nodes) and Leduc (a few thousand nodes) are small enough to build once and walk
with plain arrays, which is much faster in Python than re-simulating states each iteration.

Node kinds: TERMINAL (payoff for seat 0), CHANCE (deal or public card, with probabilities), and
DECISION (a seat to act, with its information set and legal actions). Node ids follow a
preorder walk, so every parent has a smaller id than its children; `levels` groups nodes by depth.

Information sets: nodes with the same `infoset(seat)` key share one strategy. The tree records
each information set's actions once, and the code checks that every node in a set offers the same
actions and has the same depth (true for these games, since the public history fixes the depth).

Exploitability: for a strategy profile, how much a best-responding opponent gains. For a Nash
equilibrium it is zero. Computing it exactly needs a best response per information set, and this
module does it in two passes over the tree (see `best_response`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TERMINAL, CHANCE, DECISION = 0, 1, 2


@dataclass
class GameTree:
    """Arrays describing the tree. Index every list by node id, or by information-set id where noted."""

    kind: list[int] = field(default_factory=list)
    player: list[int] = field(default_factory=list)      # decision nodes: seat to act; others -1
    info: list[int] = field(default_factory=list)        # decision nodes: information-set id; others -1
    children: list[list[int]] = field(default_factory=list)
    probs: list[list[float]] = field(default_factory=list)  # chance nodes: outcome probabilities
    util: list[float] = field(default_factory=list)      # terminal nodes: seat 0's payoff
    depth: list[int] = field(default_factory=list)
    levels: list[list[int]] = field(default_factory=list)   # node ids grouped by depth
    infosets: list[str] = field(default_factory=list)       # information-set id -> key
    info_actions: list[tuple[str, ...]] = field(default_factory=list)  # information-set id -> actions
    info_player: list[int] = field(default_factory=list)    # information-set id -> seat
    info_index: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.kind)


def build_tree(root) -> GameTree:
    """Walk every history from `root` (a game state) and record it as a node.

    `root` only needs the state protocol: is_terminal, is_chance, chance_outcomes, player, actions,
    apply, payoff, and infoset(seat). Both games in this package implement it.
    """
    tree = GameTree()

    def add(state, depth: int) -> int:
        node = len(tree.kind)
        # Placeholders keep the arrays aligned; children are filled in after the recursion below.
        tree.kind.append(DECISION)
        tree.player.append(-1)
        tree.info.append(-1)
        tree.children.append([])
        tree.probs.append([])
        tree.util.append(0.0)
        tree.depth.append(depth)
        while len(tree.levels) <= depth:
            tree.levels.append([])
        tree.levels[depth].append(node)

        if state.is_terminal():
            tree.kind[node] = TERMINAL
            tree.util[node] = float(state.payoff())
        elif state.is_chance():
            tree.kind[node] = CHANCE
            outcomes = state.chance_outcomes()
            tree.probs[node] = [p for p, _ in outcomes]
            tree.children[node] = [add(s, depth + 1) for _, s in outcomes]
        else:
            seat = state.player()
            actions = tuple(state.actions())
            key = state.infoset(seat)
            if key not in tree.info_index:
                tree.info_index[key] = len(tree.infosets)
                tree.infosets.append(key)
                tree.info_actions.append(actions)
                tree.info_player.append(seat)
            iid = tree.info_index[key]
            # Two histories that look the same to the seat must offer the same actions, or they
            # cannot share one strategy.
            assert tree.info_actions[iid] == actions, (key, actions, tree.info_actions[iid])
            tree.kind[node] = DECISION
            tree.player[node] = seat
            tree.info[node] = iid
            tree.children[node] = [add(state.apply(a), depth + 1) for a in actions]
        return node

    add(root, 0)
    # Every node of one information set sits at one depth; the best-response pass relies on it.
    depth_of_info: dict[int, int] = {}
    for n in range(tree.size):
        if tree.kind[n] == DECISION:
            d = depth_of_info.setdefault(tree.info[n], tree.depth[n])
            assert d == tree.depth[n], "information set spans two depths"
    return tree


def expected_value(tree: GameTree, sigma: list[list[float]]) -> float:
    """Seat 0's expected payoff when both seats play `sigma` (one probability list per information set)."""

    def value(n: int) -> float:
        k = tree.kind[n]
        if k == TERMINAL:
            return tree.util[n]
        if k == CHANCE:
            return sum(p * value(c) for p, c in zip(tree.probs[n], tree.children[n]))
        s = sigma[tree.info[n]]
        return sum(s[j] * value(c) for j, c in enumerate(tree.children[n]))

    return value(0)


def best_response(tree: GameTree, sigma: list[list[float]], seat: int) -> float:
    """Expected payoff to `seat` when it best-responds to the other seat's `sigma`.

    The best response has to be one choice per information set, the same at every history in it,
    so the usual recursion (pick the best child at each node) is not enough. Two passes do it:

    1. Forward (parents before children): reach[h] is the probability that the *opponent and chance*
       lead to h. The seat's own actions do not enter, because the seat is choosing them.
    2. Backward, deepest level first: at an opponent or chance node the value is the average over
       its children. At an information set of `seat`, the counterfactual value of each action is
       CFV(I, a) = sum over h in I of reach[h] * value(h.a). The set takes the action with the
       largest CFV, and every node in it takes that child's value. Information sets at one depth
       are all decided before any shallower node reads their values.
    """
    n_nodes = tree.size
    reach = [0.0] * n_nodes
    reach[0] = 1.0
    for n in range(n_nodes):
        k = tree.kind[n]
        if k == TERMINAL:
            continue
        r = reach[n]
        if k == CHANCE:
            for p, c in zip(tree.probs[n], tree.children[n]):
                reach[c] += r * p
        elif tree.player[n] != seat:
            s = sigma[tree.info[n]]
            for j, c in enumerate(tree.children[n]):
                reach[c] += r * s[j]
        else:
            for c in tree.children[n]:
                reach[c] += r  # own action: opponent reach unchanged

    sign = 1.0 if seat == 0 else -1.0
    val = [0.0] * n_nodes
    for level in reversed(tree.levels):
        own_by_info: dict[int, list[int]] = {}
        for n in level:
            k = tree.kind[n]
            if k == TERMINAL:
                val[n] = sign * tree.util[n]
            elif k == CHANCE:
                val[n] = sum(p * val[c] for p, c in zip(tree.probs[n], tree.children[n]))
            elif tree.player[n] != seat:
                s = sigma[tree.info[n]]
                val[n] = sum(s[j] * val[c] for j, c in enumerate(tree.children[n]))
            else:
                own_by_info.setdefault(tree.info[n], []).append(n)
        for nodes in own_by_info.values():
            width = len(tree.children[nodes[0]])
            cfv = [0.0] * width
            for n in nodes:
                for j, c in enumerate(tree.children[n]):
                    cfv[j] += reach[n] * val[c]
            best = max(range(width), key=lambda j: cfv[j])  # ties go to the first action
            for n in nodes:
                val[n] = val[tree.children[n][best]]
    return val[0]


def exploitability(tree: GameTree, sigma: list[list[float]]) -> tuple[float, float, float]:
    """(exploitability, seat 0's best-response value, seat 1's best-response value) for a profile.

    Each best-response value is what that seat gains by deviating, measured in its own payoff, so the
    exploitability is their average. It is zero exactly at a Nash equilibrium and never negative.
    """
    br0 = best_response(tree, sigma, 0)
    br1 = best_response(tree, sigma, 1)
    # Seat 1's best-response value is in seat 1's own payoff already (best_response negates it).
    return (br0 + br1) / 2, br0, br1
