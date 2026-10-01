from experiment.run_experiments import run


def test_experiment_report_compares_static_and_seeded_cases_to_reference(tmp_path):
    report = run(include_synthetic=True)

    assert report["overall_status"] == "success"
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["passed_cases"] >= 10
    assert all(case["status"] == "success" for case in report["cases"])
    assert any(case["source"].startswith("numpy_default_rng") for case in report["cases"])
    for case in report["cases"]:
        assert case["steps"]
        assert any(check["basis"].startswith("independent Fraction") for check in case["checks"])
