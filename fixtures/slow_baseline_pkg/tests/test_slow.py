import time

import poke


def test_slow_baseline():
    time.sleep(5)
    assert poke.identity(1) == 1
