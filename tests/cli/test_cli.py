"""Tests for the command-line workflow entry points."""

import pytest


@pytest.mark.parametrize(
    ("entry_name", "runner_name", "workflow_name"),
    [
        ("short_axis_main", "generate_short_axis", "short-axis"),
        ("bullseye_main", "generate_bullseye", "bullseye"),
        ("pipeline_main", "run_complete_pipeline", "pipeline"),
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


@pytest.mark.parametrize("entry_name", ["short_axis_main", "bullseye_main", "pipeline_main"])
def test_command_help_exits_successfully(entry_name, capsys):
    """Each command must offer argparse help without requiring a configuration."""
    from cardiac_bullseye import cli

    with pytest.raises(SystemExit) as exit_info:
        getattr(cli, entry_name)(["--help"])

    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
