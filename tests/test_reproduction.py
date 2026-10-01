from __future__ import annotations

from experiments.reproduce import run_all


def test_all_documented_reproduction_scenarios_pass() -> None:
    reports, passed = run_all()

    assert passed
    assert [report["scenario"] for report in reports] == [
        "known_offset_different_density",
        "uniform_sampling_special_case",
        "known_offset_with_long_gap",
        "constant_after_detrend",
        "no_effective_overlap",
    ]


def test_dense_and_uniform_cases_have_independent_reference_values() -> None:
    reports, _ = run_all()
    by_name = {report["scenario"]: report for report in reports}

    assert by_name["known_offset_different_density"][
        "independent_reference_correlation"
    ] is not None
    assert by_name["uniform_sampling_special_case"][
        "independent_reference_correlation"
    ] is not None
    assert by_name["known_offset_different_density"]["checks"][
        "independent_reference_correlation"
    ]
    assert by_name["uniform_sampling_special_case"]["checks"][
        "independent_reference_correlation"
    ]
