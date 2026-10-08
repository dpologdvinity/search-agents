"""Fixed-limit betting rules shared by Kuhn poker and Leduc hold'em.

A betting round is a string of action letters, one per action, in the order they were taken:

    k  check          (only when nothing is owed)
    b  bet            (opens the betting, costs one bet size)
    c  call           (matches the outstanding bet and closes the round)
    r  raise          (adds one bet size on top of the outstanding bet)
    f  fold           (gives up the pot and ends the hand)

Players alternate strictly, and the first player to act in every round is seat 0. So whose turn it
is follows from the length of the round string: an even length means seat 0 acts next.

Why one module for both games: Kuhn is a single round with one bet size and no re-raise, and
Leduc is two rounds with two bet sizes and at most two bets or raises per round. The same
rules cover both once the parameters are set.
"""

from __future__ import annotations

CHECK, BET, CALL, RAISE, FOLD = "k", "b", "c", "r", "f"


def round_closed(actions: str) -> bool:
    """True when the round is over: a call closes a bet, and two checks in a row close a round with no bet."""
    return actions.endswith(CALL) or actions == CHECK + CHECK


def raises_made(actions: str) -> int:
    """Bets and raises made so far in this round. The opening bet counts as the first raise."""
    return actions.count(BET) + actions.count(RAISE)


def legal_actions(actions: str, max_raises: int) -> tuple[str, ...]:
    """Actions the player to act may take in this round.

    Nothing owed (empty round, or the last action was a check): check or bet.
    A bet is owed (last action was a bet or raise): fold, call, and raise while under the cap.
    """
    if not actions or actions[-1] == CHECK:
        return (CHECK, BET)
    if raises_made(actions) < max_raises:
        return (FOLD, CALL, RAISE)
    return (FOLD, CALL)


def commitments(actions: str, bet_size: int) -> tuple[int, int]:
    """Chips each seat has put in during this round, after the actions so far.

    Every bet and raise is one bet size, so a raise puts the raiser at the outstanding bet plus one
    bet size. A call matches the outstanding amount. Folds stop the scan; the pot is settled elsewhere.
    """
    put = [0, 0]
    seat = 0
    for a in actions:
        if a == BET or a == RAISE:
            put[seat] = max(put) + bet_size
        elif a == CALL:
            put[seat] = max(put)
        elif a == FOLD:
            break
        seat ^= 1
    return put[0], put[1]


def folder(rounds: list[str]) -> int | None:
    """Seat that folded, or None. `rounds` is the hand split by round, each round starting with seat 0."""
    for r in rounds:
        seat = 0
        for a in r:
            if a == FOLD:
                return seat
            seat ^= 1
    return None
