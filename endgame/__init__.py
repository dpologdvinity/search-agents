"""Chess endgames solved perfectly: king and queen (KQK) and king and rook (KRK) against a lone king.

rules.py     the legal moves, the unmoves (predecessors), the attack lines, and FEN
retro.py     retrograde analysis: builds the distance-to-mate table backwards from checkmate
tablebase.py the stored table, lookups, move annotations, the playing policy, and the data files
__main__.py  the terminal game, analyze, stats, and build commands

The tables are committed under endgame/data/ (about 150 KB each). They are rebuilt with
python -m endgame build, which takes a few seconds per table.
"""
