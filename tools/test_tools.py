"""Unit tests for run_operon.py and _optuna.py."""
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from run_operon import (
    load_config,
    parse_output,
    reps_kwargs_from_cfg,
    run_once,
    run_reps,
    save,
)


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

SAMPLE_OUTPUT = textwrap.dedent("""\
    some preamble line
    iteration  r2_tr  r2_te  mae_tr  mae_te
    0          0.10   0.09   1.20    1.30
    1          0.50   0.48   0.80    0.85
    500        0.92   0.91   0.12    0.14
    x0 * x1 + 3.14
""")


def test_parse_output_last_gen_only():
    df = parse_output(SAMPLE_OUTPUT, all_gens=False)
    assert len(df) == 1
    assert float(df["r2_te"].iloc[0]) == pytest.approx(0.91)


def test_parse_output_all_gens():
    df = parse_output(SAMPLE_OUTPUT, all_gens=True)
    assert len(df) == 3
    assert list(df["iteration"].astype(int)) == [0, 1, 500]


def test_parse_output_model_expression_not_in_data():
    df = parse_output(SAMPLE_OUTPUT, all_gens=True)
    # Model expression line has wrong column count — must be excluded.
    assert "x0" not in df.values.tolist()[0]


def test_parse_output_no_header():
    df = parse_output("nothing useful here\n0.1 0.2\n", all_gens=False)
    assert df.empty


def test_parse_output_empty():
    assert parse_output("", all_gens=False).empty


def test_parse_output_last_row_is_data_not_expression():
    # If the binary emits no model expression, the last data row must be kept.
    stdout = textwrap.dedent("""\
        iteration  r2_tr  r2_te
        0          0.10   0.09
        1          0.92   0.91
    """)
    df = parse_output(stdout, all_gens=True)
    assert len(df) == 2
    df_last = parse_output(stdout, all_gens=False)
    assert float(df_last["r2_te"].iloc[0]) == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

def _make_completed_process(stdout="", returncode=0, stderr=""):
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    r.stderr = stderr
    return r


def test_run_once_success():
    with patch("subprocess.run", return_value=_make_completed_process(SAMPLE_OUTPUT)):
        df = run_once("binary", [], all_gens=False)
    assert not df.empty
    assert "r2_te" in df.columns


def test_run_once_nonzero_exit_returns_empty():
    with patch("subprocess.run", return_value=_make_completed_process("garbage", returncode=1)):
        df = run_once("binary", [], all_gens=False)
    assert df.empty


def test_run_once_timeout_returns_empty():
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("binary", 5)):
        df = run_once("binary", [], all_gens=False, timeout=5)
    assert df.empty


# ---------------------------------------------------------------------------
# run_reps
# ---------------------------------------------------------------------------

def test_run_reps_skips_empty_frames():
    outputs = [SAMPLE_OUTPUT, "", SAMPLE_OUTPUT]
    call_count = 0

    def fake_run_once(binary, args, all_gens, timeout=None):
        nonlocal call_count
        result = parse_output(outputs[call_count], all_gens)
        call_count += 1
        return result

    with patch("run_operon.run_once", side_effect=fake_run_once):
        df = run_reps("binary", [], reps=3, show_progress=False)

    assert not df.empty
    assert df["rep"].nunique() == 2        # rep 1 (empty) was skipped
    assert set(df["rep"].unique()) == {0, 2}


def test_run_reps_base_seed():
    seen_args = []

    def fake_run_once(binary, args, all_gens, timeout=None):
        seen_args.append(args)
        return parse_output(SAMPLE_OUTPUT, all_gens)

    with patch("run_operon.run_once", side_effect=fake_run_once):
        run_reps("binary", ["--foo", "bar"], reps=3, base_seed=10, show_progress=False)

    assert seen_args[0][-1] == "10"
    assert seen_args[1][-1] == "11"
    assert seen_args[2][-1] == "12"


# ---------------------------------------------------------------------------
# reps_kwargs_from_cfg
# ---------------------------------------------------------------------------

def test_reps_kwargs_default_reps_non_adaptive():
    kw = reps_kwargs_from_cfg({})
    assert kw["reps"] == 1
    assert kw["adaptive"] is False


def test_reps_kwargs_adaptive_no_reps():
    kw = reps_kwargs_from_cfg({"adaptive": True})
    assert kw["reps"] is None   # let max_reps govern
    assert kw["adaptive"] is True


def test_reps_kwargs_explicit_reps():
    kw = reps_kwargs_from_cfg({"reps": 20})
    assert kw["reps"] == 20


def test_reps_kwargs_metric_and_timeout():
    kw = reps_kwargs_from_cfg({"metric": "mae_te", "timeout": 60})
    assert kw["metric"] == "mae_te"
    assert kw["timeout"] == 60


# ---------------------------------------------------------------------------
# save / load_config
# ---------------------------------------------------------------------------

def test_save_feather_roundtrip(tmp_path):
    df = pd.DataFrame({"rep": [0, 1], "r2_te": [0.9, 0.95]})
    out = tmp_path / "out.feather"
    save(df, out)
    assert out.exists()
    loaded = pd.read_feather(out)
    assert list(loaded["r2_te"]) == pytest.approx([0.9, 0.95])


def test_save_feather_append(tmp_path):
    df1 = pd.DataFrame({"rep": [0], "r2_te": [0.9]})
    df2 = pd.DataFrame({"rep": [0], "r2_te": [0.95]})
    out = tmp_path / "out.feather"
    save(df1, out)
    save(df2, out, append=True)
    loaded = pd.read_feather(out)
    assert len(loaded) == 2
    assert list(loaded["rep"]) == [0, 1]   # rep offset applied


def test_save_csv_append(tmp_path):
    df1 = pd.DataFrame({"r2_te": [0.9]})
    df2 = pd.DataFrame({"r2_te": [0.95]})
    out = tmp_path / "out.csv"
    save(df1, out)
    save(df2, out, append=True)
    loaded = pd.read_csv(out)
    assert len(loaded) == 2


def test_load_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("binary: ./operon_gp\nreps: 10\n")
    cfg = load_config(cfg_path)
    assert cfg["binary"] == "./operon_gp"
    assert cfg["reps"] == 10


# ---------------------------------------------------------------------------
# _optuna suggest validation
# ---------------------------------------------------------------------------

def test_suggest_log_and_step_raises():
    from _optuna import suggest
    trial = MagicMock()
    with pytest.raises(ValueError, match="mutually exclusive"):
        suggest(trial, "lr", {"type": "float", "low": 1e-4, "high": 1.0, "log": True, "step": 0.1})


def test_suggest_float_no_step():
    from _optuna import suggest
    trial = MagicMock()
    trial.suggest_float.return_value = 0.5
    result = suggest(trial, "lr", {"type": "float", "low": 0.0, "high": 1.0})
    trial.suggest_float.assert_called_once_with("lr", 0.0, 1.0, log=False, step=None)
    assert result == 0.5


def test_suggest_int():
    from _optuna import suggest
    trial = MagicMock()
    trial.suggest_int.return_value = 5
    suggest(trial, "depth", {"type": "int", "low": 1, "high": 10})
    trial.suggest_int.assert_called_once_with("depth", 1, 10, step=1)


def test_suggest_categorical():
    from _optuna import suggest
    trial = MagicMock()
    trial.suggest_categorical.return_value = "add,mul"
    suggest(trial, "ops", {"type": "categorical", "choices": ["add,mul", "add,mul,div"]})
    trial.suggest_categorical.assert_called_once()
