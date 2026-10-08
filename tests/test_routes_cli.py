"""The terminal route planner: solver comparison, single runs, plots, and CSV input."""

import re

import pytest

from routes.__main__ import load_cities, main, plot
from routes.tsp import SOLVERS

# Small budgets keep each run well under a second.
FAST = ["--iterations", "300", "--population", "20", "--generations", "5"]


def test_compare_lists_every_solver(capsys):
    assert main(["--cities", "8", "--seed", "2", *FAST]) == 0
    out = capsys.readouterr().out
    for name in [*SOLVERS, "held_karp"]:
        assert any(line.startswith(name) for line in out.splitlines()), name
    assert "above optimal" in out and "best tour:" in out


def test_compare_skips_held_karp_on_a_big_map(capsys):
    assert main(["--cities", "13", *FAST]) == 0
    out = capsys.readouterr().out
    assert not re.search(r"^held_karp +[\d.]+ ", out, re.MULTILINE)  # no table row with a length
    assert "held_karp skipped" in out


def test_single_solver_prints_length_and_plot(capsys):
    assert main(["--solver", "nearest_neighbor_2opt", "--cities", "6"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("nearest_neighbor_2opt: length")
    grid = "\n".join(lines[2:])
    assert grid.startswith("+") and "|" in grid
    for label in "012345":  # every city is labelled on the map
        assert label in grid


def test_held_karp_refuses_more_than_its_limit(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--solver", "held_karp", "--cities", "13"])
    assert exit_info.value.code == 2
    assert "at most 12 cities" in capsys.readouterr().err


def test_plot_labels_every_city_and_draws_the_route():
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    text = plot(corners, [0, 1, 2, 3], width=20, height=10)
    for label in "0123":
        assert label in text
    assert "." in text


def test_load_cities_skips_header_and_blank_lines(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text("x,y\n0,0\n1,0\n\n0.5,1\n")
    assert load_cities(str(path)) == [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]


def test_load_cities_reports_a_bad_line(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text("0,0\n1,oops\n")
    with pytest.raises(ValueError, match="line 2"):
        load_cities(str(path))


def test_main_runs_on_a_csv_file(tmp_path, capsys):
    path = tmp_path / "cities.csv"
    path.write_text("x,y\n0,0\n4,0\n4,3\n0,3\n")
    assert main(["--file", str(path), "--solver", "held_karp"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("4 cities from ")
    assert "held_karp: length 14.000" in out  # the rectangle's perimeter


def test_main_rejects_too_few_cities(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--cities", "2"])
    assert exit_info.value.code == 2
    assert "at least 3 cities" in capsys.readouterr().err
