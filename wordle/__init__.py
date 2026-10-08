"""Wordle solver: pick each guess by information theory, and show how the candidates shrink.

feedback.py   the scoring rule: a guess against an answer gives five tiles, gray / yellow / green
lexicon.py    the word lists and the feedback table (every guess against every answer, precomputed)
solver.py     the strategies: maximum expected information (entropy), minimax, and random consistent
bench.py      plays every answer with each strategy and reports the guess counts
__main__.py   the terminal game: play, solve (the AI guesses), watch, and benchmark
"""
