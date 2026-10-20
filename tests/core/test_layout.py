from vision_toolkit.core.layout import Setup, load_setup, save_setup


def test_setup_round_trip(tmp_path):
    path = tmp_path / "setup.json"
    save_setup(path, Setup((0, 68, 1674, 944), 690, (1920, 1080), confirmed=True))
    assert load_setup(path) == Setup((0, 68, 1674, 944), 690, (1920, 1080), True)
    save_setup(path, Setup(None, None, (640, 400)))
    assert load_setup(path) == Setup(None, None, (640, 400), False)
