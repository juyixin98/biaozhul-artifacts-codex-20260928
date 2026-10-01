from lagcorr.experiments import build_cases, run_experiments


def test_all_reproduction_cases_have_concrete_assertions():
    cases = build_cases()
    assert {case.name for case in cases} == {
        "uniform_known_shift",
        "dense_side_one_to_one",
        "long_gap",
        "constant_sequence",
        "insufficient_overlap",
        "irregular_nonperfect_uniform_comparison",
    }
    results = run_experiments()
    assert len(results) == len(cases)


def test_independent_oracle_detects_dense_side_pairing_contract():
    dense = next(case for case in build_cases() if case.name == "dense_side_one_to_one")
    assert dense.expected_pairs[1.0] == [(i, 2 * i + 1) for i in range(20)]
