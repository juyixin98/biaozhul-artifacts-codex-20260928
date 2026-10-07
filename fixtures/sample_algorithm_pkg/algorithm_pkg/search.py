"""Search routine with a comparison that can become an infinite loop."""


def find_index(values: list[int], target: int) -> int:
    index = 0
    while index < len(values):
        if values[index] == target:
            return index
        index += 1
    return -1
