"""Tests for the command-line workflow entry points."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import pytest


@pytest.mark.parametrize(
    ("entry_name", "runner_name", "workflow_name"),
    [
        ("short_axis_main", "generate_short_axis", "short-axis"),
        ("bullseye_main", "generate_bullseye", "bullseye"),
        ("pipeline_main", "run_complete_pipeline", "pipeline"),
        ("lge_bullseye_main", "generate_lge_bullseye", "lge-bullseye"),
    ],
)
def test_command_dispatches_to_its_workflow(
    monkeypatch, tmp_path, entry_name, runner_name, workflow_name
):
    """A command must load its selected workflow and invoke its runner."""
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path, workflow: (calls.append((path, workflow)), sentinel)[1],
    )
    monkeypatch.setattr(cli, runner_name, lambda config: calls.append(config))

    assert getattr(cli, entry_name)(["--config", str(config_path)]) == 0
    assert calls == [(config_path, workflow_name), sentinel]


def test_bullseye_cli_forwards_single_reference_mode(monkeypatch, tmp_path):
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path, workflow: (calls.append((path, workflow)), sentinel)[1],
    )
    monkeypatch.setattr(
        cli,
        "generate_bullseye",
        lambda config, *, landmark_mode: calls.append((config, landmark_mode)),
    )

    assert cli.bullseye_main(
        ["--config", str(config_path), "--mode", "single-reference"]
    ) == 0
    assert calls == [
        (config_path, "bullseye"),
        (sentinel, "single-reference"),
    ]


def test_intensity_bullseye_cli_forwards_single_reference_mode(
    monkeypatch, tmp_path
):
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path, workflow: (calls.append((path, workflow)), sentinel)[1],
    )
    monkeypatch.setattr(
        cli,
        "generate_intensity_bullseye",
        lambda config, *, landmark_mode: calls.append((config, landmark_mode)),
    )

    assert cli.intensity_bullseye_main(
        ["--config", str(config_path), "--mode", "single-reference"]
    ) == 0
    assert calls == [
        (config_path, "intensity-bullseye"),
        (sentinel, "single-reference"),
    ]


@pytest.mark.parametrize(
    "options, expected",
    [([], {}),
     (["--mode", "per-slice"], {"landmark_mode": "per-slice"}),
     (["--mode", "single-reference"], {"landmark_mode": "single-reference"}),
     (["--projection", "1-4"], {"projection": "1-4"}),
     (["--projection", "0-4"], {"projection": "0-4"})]
    + [(["--mode", mode, "--projection", projection],
        {"landmark_mode": mode, "projection": projection})
       for mode in ("per-slice", "single-reference")
       for projection in ("1-4", "0-4")],
)
def test_lge_cli_forwards_requested_annotation_and_projection(
    monkeypatch, tmp_path, options, expected
):
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    sentinel = object()
    calls = []

    def load(path, workflow):
        assert path == config_path
        assert workflow == "lge-bullseye"
        return sentinel

    monkeypatch.setattr(cli, "load_config", load)
    monkeypatch.setattr(
        cli, "generate_lge_bullseye",
        lambda config, **kwargs: calls.append((config, kwargs)),
    )

    assert cli.lge_bullseye_main(["--config", str(config_path), *options]) == 0
    assert calls == [(sentinel, expected)]


@pytest.mark.parametrize("options", [["--mode", "other"], ["--projection", "2-4"]])
def test_lge_cli_rejects_invalid_modes_before_loading(monkeypatch, options):
    from cardiac_bullseye import cli

    monkeypatch.setattr(cli, "load_config", lambda *_: pytest.fail("must not load"))
    with pytest.raises(SystemExit) as error:
        cli.lge_bullseye_main(["--config", "case.toml", *options])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("entry_name", "runner_name", "workflow_name"),
    [
        ("bullseye_main", "generate_bullseye", "bullseye"),
        (
            "intensity_bullseye_main",
            "generate_intensity_bullseye",
            "intensity-bullseye",
        ),
    ],
)
def test_bullseye_commands_forward_zero_to_four_projection(
    monkeypatch, tmp_path, entry_name, runner_name, workflow_name
):
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    sentinel = object()
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path, workflow: sentinel)
    monkeypatch.setattr(
        cli,
        runner_name,
        lambda config, *, projection: calls.append((config, projection)),
    )

    assert getattr(cli, entry_name)(
        ["--config", str(config_path), "--projection", "0-4"]
    ) == 0
    assert calls == [(sentinel, "0-4")]


def test_command_returns_one_and_prints_concise_error(monkeypatch, tmp_path, capsys):
    """Configuration errors must produce a nonzero status and useful stderr."""
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda path, workflow: (_ for _ in ()).throw(ValueError("bad grid")),
    )

    assert cli.bullseye_main(["--config", str(config_path)]) == 1
    assert "bad grid" in capsys.readouterr().err


@pytest.mark.parametrize("error", [OSError("output is locked"), PermissionError("denied")])
def test_run_handles_expected_filesystem_errors_without_a_traceback(
    monkeypatch, tmp_path, capsys, error
):
    """Letting a PNG-write error escape would expose a traceback to CLI users."""
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    monkeypatch.setattr(cli, "load_config", lambda *_args: object())

    def failing_runner(_config):
        raise error

    assert cli._run("bullseye", "cardiac-bullseye", failing_runner, [
        "--config", str(config_path)
    ]) == 1
    stderr = capsys.readouterr().err
    assert str(error) in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    "entry_name",
    [
        "short_axis_main",
        "bullseye_main",
        "pipeline_main",
        "lge_bullseye_main",
        "intensity_bullseye_main",
    ],
)
def test_command_help_exits_successfully(entry_name, capsys):
    """Each command must offer argparse help without requiring a configuration."""
    from cardiac_bullseye import cli

    with pytest.raises(SystemExit) as exit_info:
        getattr(cli, entry_name)(["--help"])

    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_lge_bullseye_command_returns_one_on_validation_error(
    monkeypatch, tmp_path, capsys
):
    """External-scar validation errors propagate through the public command."""
    from cardiac_bullseye import cli

    config_path = tmp_path / "case.toml"
    config_path.touch()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_args: (_ for _ in ()).throw(ValueError("scar outside myocardium")),
    )

    assert cli.lge_bullseye_main(["--config", str(config_path)]) == 1
    assert "scar outside myocardium" in capsys.readouterr().err


def test_pyproject_registers_only_the_unified_lge_bullseye_command():
    """Installing the package must expose the unified LGE command only."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]

    assert {name: target for name, target in scripts.items() if "lge" in name} == {
        "cardiac-lge-bullseye": "cardiac_bullseye.cli:lge_bullseye_main"
    }


def test_cli_exposes_only_the_unified_lge_command_and_runner():
    from cardiac_bullseye import cli

    assert {name for name in vars(cli) if name.startswith("lge_")} == {
        "lge_bullseye_main"
    }
    assert {name for name in vars(cli) if name.startswith("generate_lge")} == {
        "generate_lge_bullseye"
    }


def test_pyproject_registers_intensity_bullseye_command():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["cardiac-intensity-bullseye"] == (
        "cardiac_bullseye.cli:intensity_bullseye_main"
    )
