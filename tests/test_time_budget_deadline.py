"""A62: a time-matched read is not started when it would predictably overshoot."""
import time
from types import SimpleNamespace

from nanoqc.experiments.research_ablation import _time_budget_counts


class _Sampler:
    site_to_variables = [[0, 1]]

    def __init__(self, durations):
        self.durations = list(durations)
        self.calls = 0

    def simulated_annealing(self, num_reads, site_passes, seed):
        time.sleep(self.durations[min(self.calls, len(self.durations) - 1)])
        self.calls += 1
        return SimpleNamespace(counts={(1, 0): 1})


def test_reads_stop_before_a_predictable_overshoot():
    sampler = _Sampler([0.03])
    counts, elapsed, queries = _time_budget_counts(sampler, "sa", 0.10, 0, 1, 1)
    assert sampler.calls == 3            # a 4th read would end near 0.12 s
    assert elapsed <= 0.10 * 1.10
    assert sum(counts.values()) == 3


def test_first_read_always_runs_and_its_overrun_is_reported():
    sampler = _Sampler([0.05])
    counts, elapsed, _ = _time_budget_counts(sampler, "sa", 0.01, 0, 1, 1)
    assert sampler.calls == 1 and elapsed > 0.01


def test_slow_setup_read_leaves_the_window():
    sampler = _Sampler([0.06] + [0.005] * 100)
    _, elapsed, _ = _time_budget_counts(sampler, "sa", 0.20, 0, 1, 1)
    assert sampler.calls > 10 and elapsed <= 0.20 * 1.10
