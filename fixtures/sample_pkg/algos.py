"""Small algorithm package used as the mutation-testing fixture.

Every mutation point in this file has a hand-derived expected fate,
recorded in tests/test_runner_fates.py.
"""


def is_positive(x):
    return x > 0


def clamp_lo(x, lo):
    if x < lo:
        return lo
    return x


def sum_list(nums):
    total = 0
    for n in nums:
        total = total + n
    return total


def square(x):
    return x * x


def count_adults(ages):
    count = 0
    for age in ages:
        if age >= 18:
            count = count + 1
    return count


def first_negative_index(nums):
    i = 0
    while i < len(nums):
        if nums[i] < 0:
            return i
        i = i + 1
    return -1


def dead_code_demo(x):
    return x
    y = x + 1
    if y > 10:
        return y
