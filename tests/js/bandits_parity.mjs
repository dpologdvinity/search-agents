// Parity runner for web/js/bandits-core.js. Reads a JSON request on stdin and prints each agent's play on
// stdout, so tests/test_bandits_parity.py can compare the browser port with the Python package.
//
// Request:  {"runs": [{key, machines: {kind, k, seed, pulls, period, schedule}}]}
// Response: {"runs": [{arms, rewards, regret, optimal, reasons}]}
// Run by hand with: node tests/js/bandits_parity.mjs < request.json

import { Casino, Racer } from '../../web/js/bandits-core.js';

let input = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) input += chunk;
const req = JSON.parse(input);

const runs = req.runs.map((c) => {
  const casino = new Casino(c.machines);
  const racer = new Racer(c.key, casino);
  const reasons = [];
  while (!racer.done) reasons.push(racer.step().reason);
  return {
    arms: racer.arms,
    rewards: racer.rewards,
    regret: racer.regret,
    optimal: racer.optimal,
    reasons,
  };
});

process.stdout.write(JSON.stringify({ runs }));
