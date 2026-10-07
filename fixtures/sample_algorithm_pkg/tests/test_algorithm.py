from algorithm_pkg import boot_sentinel, classify_size, discounted_total, find_index
from algorithm_pkg.sentinel import _invariant


def test_discounted_total():
    assert discounted_total(10, 5) == 50
    assert discounted_total(20, 10) == 190


def test_find_index_found_and_missing():
    assert find_index([2, 4], 4) == 1
    assert find_index([2, 4], 9) == -1


def test_classify_size():
    assert classify_size(11) == "large"
    assert classify_size(2) == "small"


def test_sentinel_invariant():
    assert _invariant() is True
    assert boot_sentinel() == "ready"
