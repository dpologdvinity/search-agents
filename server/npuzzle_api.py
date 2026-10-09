"""N-Puzzle API: metadata, scrambles, and a WebSocket that streams a search as it runs.

WebSocket protocol (/ws/npuzzle):
  client -> {"board": [...], "algorithm": "astar", "heuristic": "manhattan", ...}
  server -> {"type": "start", ...}
            {"type": "progress", "nodes": ..., "frontier": ..., "elapsed": ...,
             "board": [...], "expanded": [[board, g, action], ...]}   (repeated)
            {"type": "result", "status": ..., "path": [...], ...}
         or {"type": "error", "message": ...}
  client -> {"type": "cancel"} stops the search early.

"expanded" carries the first TRACE_LIMIT expansions in order, so the client
can animate the search. `action` is the move that produced the board; for
bidirectional BFS, actions from the goal side are prefixed with "goal:".
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError, field_validator

from npuzzle import ALGORITHMS, HEURISTICS, SearchLimits, goal, is_solvable, successors
from npuzzle.batched import batch_heuristic, batch_weighted_a_star

router = APIRouter()

TRACE_LIMIT = 3000
PROGRESS_INTERVAL = 0.05  # seconds between progress messages
MAX_SECONDS = 10.0
# Searches that store every generated state use about 0.6-1.2 KB per
# expanded node, so they get a lower cap to bound memory. IDS and IDA*
# keep only the current path and are limited by time instead.
MAX_NODES = 250_000
MAX_NODES_LINEAR_MEMORY = 3_000_000
LINEAR_MEMORY = {"ids", "idastar"}


def node_limit(algorithm: str) -> int:
    return MAX_NODES_LINEAR_MEMORY if algorithm in LINEAR_MEMORY else MAX_NODES

# Algorithms that take a heuristic argument.
INFORMED = {"greedy", "astar", "wastar", "idastar", "bwas"}
FOUR_BY_FOUR_ONLY = {"pdb", "neural"}

DESCRIPTIONS = {
    "bfs": "Breadth-first search. Optimal; explores layer by layer.",
    "dfs": "Depth-first search. Finds a path fast but rarely a short one.",
    "ids": "Iterative deepening DFS. Optimal with DFS-sized memory.",
    "ucs": "Uniform-cost search. Optimal; orders by path cost g.",
    "bibfs": "Bidirectional BFS. Searches from both ends and meets in the middle.",
    "greedy": "Greedy best-first. Follows the heuristic h alone; not optimal.",
    "astar": "A*. Orders by f = g + h; optimal with an admissible h.",
    "wastar": "Weighted A*. f = g + w*h; faster, at most w times optimal.",
    "idastar": "IDA*. A* with iterative deepening on f; linear memory.",
    "bwas": "Batch weighted A*. Scores many states per heuristic call; built for the neural heuristic.",
}
HEURISTIC_DESCRIPTIONS = {
    "manhattan": "Sum of tile distances from their goal squares.",
    "linear_conflict": "Manhattan plus 2 for each tile that must leave its row or column.",
    "pdb": "Additive 5-5-5 pattern database: exact costs for three tile groups (4x4).",
    "neural": "Learned: pattern database plus a neural correction (4x4, not admissible).",
}


class SolveRequest(BaseModel):
    board: list[int] = Field(max_length=16)  # the largest board is 4x4; valid_board() then checks 9 or 16
    algorithm: Literal["bfs", "dfs", "ids", "ucs", "bibfs", "greedy", "astar", "wastar", "idastar", "bwas"]
    heuristic: Literal["manhattan", "linear_conflict", "pdb", "neural"] = "manhattan"
    weight: float = Field(2.0, ge=1.0, le=10.0, description="h weight for wastar")
    g_weight: float = Field(0.8, ge=0.1, le=1.0, description="g weight for bwas")
    batch_size: int = Field(100, ge=1, le=5000)
    max_seconds: float = Field(MAX_SECONDS, gt=0, le=MAX_SECONDS)

    @field_validator("board")
    @classmethod
    def valid_board(cls, board):
        n = math.isqrt(len(board))
        if n not in (3, 4) or n * n != len(board):
            raise ValueError("board must have 9 or 16 tiles")
        if sorted(board) != list(range(n * n)):
            raise ValueError("board must contain each tile 0..n*n-1 once")
        return board

    @property
    def n(self):
        return math.isqrt(len(self.board))


class Cancelled(Exception):
    pass


def make_solver(req: SolveRequest):
    """Return solve(limits, on_expand) for a validated request."""
    n = req.n
    if req.algorithm in INFORMED and req.heuristic in FOUR_BY_FOUR_ONLY and n != 4:
        raise ValueError(f"the {req.heuristic} heuristic needs a 4x4 board")
    if req.algorithm == "bwas":
        if req.heuristic == "linear_conflict":
            raise ValueError("bwas supports manhattan, pdb, and neural heuristics")
        h = batch_heuristic(req.heuristic, n)
        return lambda start, limits, cb: batch_weighted_a_star(
            start, h, batch_size=req.batch_size, g_weight=req.g_weight, limits=limits, on_expand=cb)
    if req.heuristic == "neural" and req.algorithm in INFORMED:
        raise ValueError("the neural heuristic runs with bwas (batched) only")
    fn = ALGORITHMS[req.algorithm]
    if req.algorithm == "wastar":
        h = HEURISTICS[req.heuristic]
        return lambda start, limits, cb: fn(start, heuristic=h, weight=req.weight, limits=limits, on_expand=cb)
    if req.algorithm in INFORMED:
        h = HEURISTICS[req.heuristic]
        return lambda start, limits, cb: fn(start, heuristic=h, limits=limits, on_expand=cb)
    return lambda start, limits, cb: fn(start, limits=limits, on_expand=cb)


@router.get("/api/npuzzle/meta")
async def meta():
    return {
        "algorithms": [{"name": k, "description": v, "informed": k in INFORMED} for k, v in DESCRIPTIONS.items()],
        "heuristics": [{"name": k, "description": v, "sizes": [4] if k in FOUR_BY_FOUR_ONLY else [3, 4]}
                       for k, v in HEURISTIC_DESCRIPTIONS.items()],
        "limits": {"max_seconds": MAX_SECONDS, "max_nodes": MAX_NODES,
                   "max_nodes_linear_memory": MAX_NODES_LINEAR_MEMORY, "trace_limit": TRACE_LIMIT},
    }


@router.get("/api/npuzzle/scramble")
async def scramble(n: int = 3, moves: int | None = None, seed: int | None = None):
    """A solvable board: a random walk of `moves` from the goal, or uniformly random if omitted."""
    if n not in (3, 4):
        raise HTTPException(400, "n must be 3 or 4")
    rng = random.Random(seed)
    if moves is None:
        while True:
            board = list(range(n * n))
            rng.shuffle(board)
            if is_solvable(tuple(board), n):
                return {"board": board}
    if not 0 <= moves <= 500:
        raise HTTPException(400, "moves must be in 0..500")
    board, prev = goal(n), None
    for _ in range(moves):
        options = [b for _, b in successors(board, n) if b != prev]
        prev, board = board, rng.choice(options)
    return {"board": list(board)}


async def solve_stream(ws: WebSocket, req: SolveRequest):
    """Run one search in a worker thread and stream its progress to ws."""
    solver = make_solver(req)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    trace: list = []
    stats = {"nodes": 0, "last": 0.0, "sent": 0}
    start_time = time.perf_counter()

    def on_expand(node, frontier_size):
        if cancel.is_set():
            raise Cancelled
        stats["nodes"] += 1
        if len(trace) < TRACE_LIMIT:
            trace.append([list(node.board), node.depth, node.action or "Initial"])
        now = time.perf_counter()
        if now - stats["last"] >= PROGRESS_INTERVAL:
            stats["last"] = now
            new = trace[stats["sent"]:]
            stats["sent"] = len(trace)
            msg = {"type": "progress", "nodes": stats["nodes"], "frontier": frontier_size,
                   "elapsed": round(now - start_time, 3), "board": list(node.board), "expanded": new}
            loop.call_soon_threadsafe(queue.put_nowait, msg)

    def run():
        limits = SearchLimits(max_nodes=node_limit(req.algorithm), max_seconds=req.max_seconds)
        return solver(tuple(req.board), limits, on_expand)

    async def forward():
        # The only task that sends during the search; None ends it.
        while (msg := await queue.get()) is not None:
            await ws.send_json(msg)

    async def watch_client():
        # A cancel message or a closed socket stops the search.
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    break
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        cancel.set()

    await ws.send_json({"type": "start", "algorithm": req.algorithm, "heuristic": req.heuristic,
                        "n": req.n, "trace_limit": TRACE_LIMIT})
    search = asyncio.create_task(asyncio.to_thread(run))
    sender = asyncio.create_task(forward())
    watcher = asyncio.create_task(watch_client())
    try:
        result = await search
    except Cancelled:
        result = None
    finally:
        cancel.set()
        watcher.cancel()
        # Progress queued by the search thread is already ahead of this
        # sentinel, so the sender drains it in order before stopping.
        queue.put_nowait(None)
        await sender

    if result is None:
        await ws.send_json({"type": "result", "status": "cancelled", "expanded": stats["nodes"]})
        return
    tail = trace[stats["sent"]:]
    if tail:
        await ws.send_json({"type": "progress", "nodes": stats["nodes"], "frontier": 0,
                            "elapsed": round(time.perf_counter() - start_time, 3),
                            "board": tail[-1][0], "expanded": tail})
    await ws.send_json({"type": "result", **asdict(result)})


def parse_request(data) -> SolveRequest:
    try:
        req = SolveRequest.model_validate(data)
        make_solver(req)  # surfaces unsupported combinations before starting
        return req
    except ValidationError as e:
        raise ValueError("; ".join(err["msg"] for err in e.errors())) from None
