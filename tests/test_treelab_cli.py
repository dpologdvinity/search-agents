"""Command line for the tree lab: the printed report has the right parts and is the same on every run."""

import re

import pytest

from treelab.cli import main


def test_grow_prints_rules_accuracy_and_region_map(capsys):
    main(["grow", "--data", "xor", "--depth", "4"])
    out = capsys.readouterr().out
    assert "yes:" in out and "no:" in out
    assert "train accuracy" in out and "test accuracy" in out
    assert out.count("|") >= 2 * 20  # the 20-row region map
    assert re.search(r"leaves \d+", out)


def test_grow_prune_reports_both_leaf_counts(capsys):
    main(["grow", "--data", "nested", "--depth", "6", "--prune", "0.02"])
    out = capsys.readouterr().out
    assert re.search(r"leaves \d+  \(unpruned \d+\)", out)


def test_forest_prints_oob_test_and_importances(capsys):
    main(["forest", "--data", "spiral", "--trees", "10", "--depth", "4"])
    out = capsys.readouterr().out
    assert "OOB accuracy" in out and "test accuracy" in out
    assert "feature importances" in out
    assert "x^2+y^2" in out


def test_output_is_deterministic(capsys):
    main(["grow", "--data", "diagonal", "--seed", "3"])
    first = capsys.readouterr().out
    main(["grow", "--data", "diagonal", "--seed", "3"])
    assert capsys.readouterr().out == first


def test_bad_preset_is_rejected():
    with pytest.raises(SystemExit):
        main(["grow", "--data", "moons"])


@pytest.mark.parametrize(
    "argv",
    [
        ["grow", "--n", "0"],
        ["grow", "--test", "0"],
        ["grow", "--test", "1.0"],
        ["grow", "--test", "nan"],
        ["grow", "--depth", "-1"],
        ["grow", "--min-leaf", "0"],
        ["grow", "--prune", "-0.1"],
        ["forest", "--trees", "0"],
    ],
)
def test_out_of_range_settings_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "must be" in capsys.readouterr().err
