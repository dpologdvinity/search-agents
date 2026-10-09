"""Lost Robot: where am I on a known floor plan? Bayes filtering with a histogram grid and with particles.

prng.py       the seeded random stream shared with the JavaScript port (mulberry32, the generator of bandits/rng.py)
world.py      the floor plans, the odometry and range-sensor models, and the ray caster the filters reuse
grid.py       histogram (grid) Bayes filter over x, y and heading: exact on its discretisation
particles.py  Monte Carlo localisation with systematic resampling and augmented-MCL kidnap recovery
sim.py        the true robot, a route-following autopilot, and the mode count shown on the page
benchmark.py  seeded trials: convergence steps, failure rate, particle counts, noise levels, kidnap recovery
"""
