import importlib.util


def test_complete_pipeline_module_and_public_api_are_removed():
    import cardiac_bullseye.workflows as workflows

    assert importlib.util.find_spec("cardiac_bullseye.workflows.complete") is None
    assert not hasattr(workflows, "run_complete_pipeline")
