"""Ghost Hunt: track invisible ghosts from noisy sonar, with a hidden Markov model over their positions.

maze.py         a seeded perfect maze, its open cells and all-pairs distances
motion.py       the ghost motion models (random, lurker, patrol) as known transition distributions
sonar.py        the noisy distance reading and its discrete Gaussian likelihood
hmm.py          exact inference: the forward algorithm (filtering) and Viterbi (the most likely path)
particles.py    the particle filter approximation of the same belief
game.py         the board: ghosts, the player, the turn rules and each ghost's filter
agent.py        an autopilot that trusts the belief, walks to the likeliest cell and busts when sure
benchmark.py   seeded benchmark: turns to bust, belief sharpness, Viterbi accuracy and particle convergence
"""
