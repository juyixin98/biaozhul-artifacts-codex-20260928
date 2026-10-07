import algos


def test_is_positive():
    assert algos.is_positive(5) is True
    assert algos.is_positive(0) is False
    assert algos.is_positive(-3) is False


def test_clamp_lo():
    assert algos.clamp_lo(3, 0) == 3
    assert algos.clamp_lo(-5, 0) == 0


def test_sum_list():
    assert algos.sum_list([1, 2, 3]) == 6
    assert algos.sum_list([]) == 0


def test_square():
    assert algos.square(4) == 16


def test_count_adults():
    assert algos.count_adults([20, 10, 35]) == 2
    assert algos.count_adults([]) == 0


def test_first_negative_index():
    assert algos.first_negative_index([3, -2, 5]) == 1
    assert algos.first_negative_index([1, 2]) == -1
    assert algos.first_negative_index([0, -1]) == 1


def test_dead_code_demo():
    assert algos.dead_code_demo(7) == 7
