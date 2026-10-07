import calc


def test_add_wrong_expectation():
    # Deliberately wrong: the unmutated suite is already red.
    assert calc.add(1, 1) == 3
