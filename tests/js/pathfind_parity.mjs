// Parity runner for web/js/pathfind-core.js. Reads a JSON request on stdin and prints the results on
// stdout, so tests/test_pathfind_parity.py can compare the browser port with the Python package.
//
// Request:  {"rng": [seed, ...], "u32": 8, "runs": [{maze, size, seed, density, swamp, algo, diagonal, heur, weight}]}
// Response: {"rng": [[u32, ...], ...], "runs": [{cells, start, goal, found, path, cost, steps, generated}]}
// Run by hand with: node tests/js/pathfind_parity.mjs < request.json

import { Rng, makeMaze, run } from '../../web/js/pathfind-core.js';

let input = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) input += chunk;
const req = JSON.parse(input);

const rng = req.rng.map((seed) => {
  const r = new Rng(seed);
  return Array.from({ length: req.u32 }, () => r.u32());
});

const runs = req.runs.map((c) => {
  const m = makeMaze(c.maze, c.size, c.size, c.seed, { density: c.density, swamp: c.swamp });
  const res = run(c.algo, m.grid, m.start, m.goal, { diagonal: c.diagonal, heur: c.heur, weight: c.weight });
  return {
    cells: m.grid.cells,
    start: m.start,
    goal: m.goal,
    found: res.found,
    path: res.path,
    cost: res.cost,
    steps: res.steps,
    generated: res.generated,
  };
});

process.stdout.write(JSON.stringify({ rng, runs }));
