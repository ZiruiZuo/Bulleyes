from cardiac_bullseye.short_axis import planner


def test_interactive_backend_check_runs_with_a_safe_backend_switch(monkeypatch):
    """The public planner can evaluate a headless Matplotlib backend."""

    planner.plt.get_backend()
    monkeypatch.setattr(planner.plt, "switch_backend", lambda _backend: None)

    planner._ensure_interactive_backend()
