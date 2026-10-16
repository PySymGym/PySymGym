"""Tests for the per-trial artifact directory layout."""

from pathlib import Path

from paths import REPORT_PATH, TRIALS_PATH, trial_dir


class TestTrialDir:
    def test_layout(self):
        assert trial_dir(3) == REPORT_PATH / "trials" / "3"

    def test_is_under_trials_path(self):
        assert trial_dir(0).is_relative_to(TRIALS_PATH)

    def test_distinct_per_trial_number(self):
        dirs = [trial_dir(n) for n in range(5)]
        assert len(set(dirs)) == 5

    def test_is_a_path(self):
        assert isinstance(trial_dir(1), Path)
