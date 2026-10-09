Command: `PYTHONPATH=. .venv/bin/python -c 'import json, random, statistics, sys; from checkers.board import START, RED, legal_moves, notation; from checkers.search import AlphaBeta, Minimax; rows = []\nfor seed in range(1, 11):\n    rng = random.Random(seed); board, side = START, RED\n    for _ in range(8):\n        board, side = rng.choice(legal_moves(board, side)).result, -side\n    forced = len(legal_moves(board, side)) == 1\n    row = {"seed": seed, "forced_root": forced}\n    for d in (4, 5, 6):\n        m = Minimax(d).search(board, side); a = AlphaBeta(max_depth=d, max_seconds=1e9).search(board, side)\n        root = {notation(x): s for x, s in m.root}\n        row["depth%d" % d] = {"minimax_nodes": m.nodes, "alphabeta_nodes": a.nodes, "alphabeta_depth": a.depth, "minimax_move": notation(m.move), "alphabeta_move": notation(a.move), "minimax_score": m.score, "alphabeta_score": a.score, "same_score": m.score == a.score, "same_move": notation(m.move) == notation(a.move), "alphabeta_move_minimax_score": root.get(notation(a.move))}\n    rows.append(row)\ndef tally(d):\n    k = "depth%d" % d\n    free = [r for r in rows if not r["forced_root"]]\n    mism = [r[k] for r in free if not r[k]["same_move"]]\n    return {"minimax_nodes_median": statistics.median(r[k]["minimax_nodes"] for r in rows), "alphabeta_nodes_median": statistics.median(r[k]["alphabeta_nodes"] for r in rows), "same_score_nonforced": "%d of %d" % (sum(r[k]["same_score"] for r in free), len(free)), "same_move_nonforced": "%d of %d" % (sum(r[k]["same_move"] for r in free), len(free)), "move_mismatches_all_ties": all(x["alphabeta_move_minimax_score"] == x["minimax_score"] for x in mism)}\nprint(json.dumps({"command": "PYTHONPATH=. .venv/bin/python -c " + repr(sys.argv[1]), "position": "8 random legal plies from the opening: random.Random(seed).choice over legal_moves at each ply; red to move after 8 plies; seeds 1-10", "minimax": "Minimax(d): fixed depth d, no pruning, node count = _negamax calls", "alphabeta": "AlphaBeta(max_depth=d, max_seconds=1e9): iterative deepening, node count summed over depths 1..d (what the agent reports)", "notes": "forced_root positions (one legal move) use AlphaBeta forced_depth=4 scoring, so their scores are not comparable to depth d", "summary": {"depth%d" % d: tally(d) for d in (4, 5, 6)}, "rows": rows}, indent=1))'`

8 random legal plies from the opening: random.Random(seed).choice over legal_moves at each ply; red to move after 8 plies; seeds 1-10.

forced_root positions (one legal move) use AlphaBeta forced_depth=4 scoring, so their scores are not comparable to depth d.

Node counts (minimax / alpha-beta) by seed, same root move and score unless noted:

| Seed | Forced root | Depth 4 | Depth 5 | Depth 6 |
|---|---|---|---|---|
| 1 | no | 2,962 / 618 | 16,845 / 1,252 | 92,608 / 3,566 |
| 2 | yes | 112 / 29 | 810 / 29 | 4,373 / 29 (score differs) |
| 3 | no | 4,445 / 602 | 28,472 / 1,569 | 187,646 / 4,915 |
| 4 | no | 4,395 / 496 | 25,107 / 1,364 (move tie differs) | 147,554 / 2,169 |
| 5 | no | 2,662 / 533 | 14,222 / 1,349 | 44,216 / 3,180 |
| 6 | no | 3,784 / 437 | 14,237 / 1,322 | 87,537 / 3,117 |
| 7 | no | 740 / 253 (move tie differs) | 4,030 / 591 | 21,622 / 2,235 |
| 8 | no | 9,362 / 598 | 75,915 / 4,209 | 395,831 / 6,170 (move tie differs) |
| 9 | no | 5,482 / 500 | 24,840 / 1,199 | 145,757 / 1,604 |
| 10 | yes | 536 / 266 | 3,793 / 266 (score differs) | 28,084 / 266 (score differs) |

Medians over the 10 positions and agreement (non-forced roots):

| Depth | Minimax median | Alpha-beta median | Same score | Same move | Move mismatches are ties |
|---|---|---|---|---|---|
| 4 | 3,373 | 498 | 8 of 8 | 7 of 8 | yes |
| 5 | 15,541 | 1,287 | 8 of 8 | 7 of 8 | yes |
| 6 | 90,072 | 2,676 | 8 of 8 | 7 of 8 | yes |

Perft from the opening (depths 1-5): 7, 49, 302, 1,469, 7,361 (command: PYTHONPATH=. .venv/bin/python -c "from checkers.board import START, RED, perft; print([perft(START, RED, d) for d in range(1, 6)])").

