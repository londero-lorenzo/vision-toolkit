import numpy as np

from vision_toolkit.usecases.anonymize.doubts import episodes, representative, select, unexplained_blobs


def test_episodes_group_close_times():
    times = [10, 0, 1, 2, 11, 30]
    assert episodes(times, 2) == [[0, 1, 2], [10, 11], [30]]
    assert episodes(times, 0) == [[0], [1], [2], [10], [11], [30]]
    rows = [{"t": 5}, {"t": 4}]
    assert episodes(rows, 1, key=lambda row: row["t"]) == [[{"t": 4}, {"t": 5}]]
    assert representative([0, 1, 2]) == 1 and representative([10, 11]) == 11


def test_select_keeps_everything_without_a_budget():
    groups = [[t] for t in range(5)]
    assert select(groups) == groups
    assert select(groups, budget=10) == groups


def test_select_spreads_a_budget_over_time():
    groups = [[t] for t in range(5)]
    assert select(groups, budget=2) == [[0], [4]]
    assert select(groups, budget=3) == [[0], [2], [4]]
    assert select(groups, budget=1) == [[2]]
    assert select(groups, budget=0) == []


def test_unexplained_blobs():
    foreground = np.zeros((100, 100), np.uint8)
    foreground[10:30, 10:30] = 255   # explained by a pose
    foreground[60:80, 60:80] = 255   # nobody explains it
    foreground[5:8, 90:93] = 255     # too small to matter
    explained = np.zeros_like(foreground)
    explained[0:40, 0:40] = 255
    assert unexplained_blobs(foreground, explained, min_area=50) == [(60, 60, 80, 80)]
