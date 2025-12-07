def test_infer_import():
    from src.infer import run_inference
    assert callable(run_inference)
