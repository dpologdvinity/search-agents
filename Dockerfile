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
COPY markov ./markov
COPY clusters ./clusters
COPY nnlab ./nnlab
COPY pathfind ./pathfind
COPY mdplab ./mdplab
COPY optlab ./optlab
COPY treelab ./treelab
COPY cartpole ./cartpole
COPY queens ./queens
COPY snake ./snake
COPY rover ./rover
COPY ghosthunt ./ghosthunt
COPY tetris ./tetris
COPY nonogram ./nonogram
COPY regression ./regression
COPY localize ./localize
COPY walkers ./walkers
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
COPY tictactoe ./tictactoe
COPY server ./server
COPY web ./web

RUN useradd --create-home app
USER app

EXPOSE 8080
# PORT and FORWARDED_ALLOW_IPS default to Fly: only Fly's proxy (172.16.0.0/12 or fdaa::/16, see server/limits.py) may
# set X-Forwarded-For. Another host sets them to its own port and proxy range (Hugging Face Spaces: 7860, 10.0.0.0/8).
# uvicorn takes the client from the rightmost X-Forwarded-For entry outside the trusted ranges, so a client cannot
# spoof its address by sending the header itself. WebSocket messages are capped at 256 KB, the same as HTTP bodies,
# instead of uvicorn's 16 MB default.
ENV PORT=8080 FORWARDED_ALLOW_IPS=172.16.0.0/12,fdaa::/16
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port \"$PORT\" --proxy-headers --forwarded-allow-ips \"$FORWARDED_ALLOW_IPS\" --ws-max-size 262144"]
