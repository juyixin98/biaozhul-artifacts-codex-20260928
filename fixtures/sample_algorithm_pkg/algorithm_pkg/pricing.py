"""Arithmetic and constant-fold fixture functions."""


EQUIVALENT_CONSTANTS = 3 * 1


def discounted_total(unit_price: float, quantity: int) -> float:
    subtotal = unit_price * quantity
    if subtotal > 100:
        return subtotal - 10
    return subtotal


def spread(a: float, b: float) -> float:
    return a - b
