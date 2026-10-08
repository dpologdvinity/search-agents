# Serving image: search code, data files, backend, and frontend. No torch:
# the neural heuristic runs from exported NumPy weights.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir "numpy>=1.26" "fastapi>=0.115" "uvicorn[standard]>=0.30"

COPY npuzzle ./npuzzle
COPY connect4 ./connect4
COPY checkers ./checkers
COPY pacman ./pacman
COPY battleship ./battleship
COPY blackjack ./blackjack
COPY lightsout ./lightsout
COPY bandits ./bandits
COPY cartpole ./cartpole
COPY queens ./queens
COPY snake ./snake
COPY rover ./rover
COPY tetris ./tetris
COPY nonogram ./nonogram
COPY endgame ./endgame
COPY sokoban ./sokoban
COPY wordle ./wordle
COPY poker ./poker
COPY minesweeper ./minesweeper
COPY hexgame ./hexgame
COPY routes ./routes
COPY warehouse ./warehouse
COPY game2048 ./game2048
COPY sudoku ./sudoku
COPY server ./server
COPY web ./web
COPY results/bandits_benchmark.json ./results/bandits_benchmark.json

RUN useradd --create-home app
USER app

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
