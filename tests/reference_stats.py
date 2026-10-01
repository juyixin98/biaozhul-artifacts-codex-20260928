"""Independent stdlib-only reference implementation used by tests only.

This module deliberately imports neither NumPy nor the application kernel.
Floating-point sums use Python's exact integer/sum operations on floats, giving
an independently ordered comparison for the NumPy implementation.
"""

from __future__ import annotations


def validate(p, y, w):
    if not (len(p) == len(y) == len(w)):
        raise ValueError("length mismatch")
    if not p:
        raise ValueError("empty input")
    for value in p:
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability out of range")
    for value in y:
        if value not in (0, 1):
            raise ValueError("binary outcome required")
    for value in w:
        if value <= 0.0:
            raise ValueError("weight must be positive")


def fixed_diagnostics(p, y, w, edges):
    validate(p, y, w)
    bins = [[] for _ in range(len(edges) - 1)]
    for i, probability in enumerate(p):
        assigned = None
        for index in range(len(edges) - 1):
            lower = edges[index]
            upper = edges[index + 1]
            if index == len(edges) - 2:
                inside = lower <= probability <= upper
            else:
                inside = lower <= probability < upper
            if inside:
                assigned = index
                break
        if assigned is None:
            raise ValueError("out of fixed edges")
        bins[assigned].append(i)
    return _diagnostics(p, y, w, bins)


def quantile_diagnostics(p, y, w, n_bins):
    validate(p, y, w)
    values = sorted(set(p))
    groups = {value: [] for value in values}
    weights_by_group = {value: 0.0 for value in values}
    for index, probability in enumerate(p):
        groups[probability].append(index)
        weights_by_group[probability] += w[index]

    total = sum(w)
    cumulative = [0.0]
    running = 0.0
    for value in values:
        running += weights_by_group[value]
        cumulative.append(running)

    cuts = []
    previous = 0
    group_count = len(values)
    for b in range(1, n_bins):
        target = b * total / n_bins
        best = previous
        best_distance = abs(cumulative[previous] - target)
        for boundary in range(previous, group_count + 1):
            distance = abs(cumulative[boundary] - target)
            if distance <= best_distance:
                best = boundary
                best_distance = distance
        best = max(best, previous)
        cuts.append(best)
        previous = best

    bins = []
    for index in range(n_bins):
        lower_group = 0 if index == 0 else cuts[index - 1]
        upper_group = cuts[index] if index < n_bins - 1 else group_count
        members = []
        for group_index in range(lower_group, upper_group):
            members.extend(groups[values[group_index]])
        bins.append(members)
    return _diagnostics(p, y, w, bins)


def _diagnostics(p, y, w, bins):
    total_weight = sum(w)
    total_count = len(p)
    ess = total_weight**2 / sum(weight**2 for weight in w)
    outcome_rate = sum(w[i] * y[i] for i in range(total_count)) / total_weight
    mean_p = sum(w[i] * p[i] for i in range(total_count)) / total_weight
    brier = sum(w[i] * (p[i] - y[i]) ** 2 for i in range(total_count)) / total_weight

    reliability = 0.0
    resolution = 0.0
    within_variance = 0.0
    bin_records = []
    weight_sum = 0.0
    count_sum = 0

    for members in bins:
        count = len(members)
        bin_weight = sum(w[i] for i in members)
        weight_sum += bin_weight
        count_sum += count
        if bin_weight == 0.0:
            bin_records.append(
                {
                    "count": 0,
                    "weight": 0.0,
                    "ess": 0.0,
                    "empty": True,
                    "p_mean": None,
                    "y_mean": None,
                }
            )
            continue
        p_mean = sum(w[i] * p[i] for i in members) / bin_weight
        y_mean = sum(w[i] * y[i] for i in members) / bin_weight
        bin_ess = bin_weight**2 / sum(w[i] ** 2 for i in members)
        reliability += bin_weight / total_weight * (p_mean - y_mean) ** 2
        resolution += bin_weight / total_weight * (y_mean - outcome_rate) ** 2
        within_variance += bin_weight / total_weight * sum(
            w[i] * (p[i] - p_mean) ** 2 for i in members
        ) / bin_weight
        bin_records.append(
            {
                "count": count,
                "weight": bin_weight,
                "ess": bin_ess,
                "empty": False,
                "p_mean": p_mean,
                "y_mean": y_mean,
            }
        )

    uncertainty = outcome_rate * (1.0 - outcome_rate)
    relaxed = reliability - resolution + uncertainty
    return {
        "total_weight": total_weight,
        "total_count": total_count,
        "ess": ess,
        "outcome_rate": outcome_rate,
        "mean_probability": mean_p,
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "within_variance": within_variance,
        "relaxed_components": relaxed,
        "residual": brier - within_variance - relaxed,
        "weight_sum": weight_sum,
        "count_sum": count_sum,
        "bins": bin_records,
    }


def weighted_pava_transform(fit_p, fit_y, fit_w, eval_p):
    ordered = sorted(zip(fit_p, fit_y, fit_w), key=lambda row: row[0])
    groups = []
    for probability, outcome, weight in ordered:
        if groups and groups[-1][0] == probability:
            _, old_w, old_events = groups[-1]
            new_w = old_w + weight
            groups[-1] = (probability, new_w, old_events + outcome * weight)
        else:
            groups.append((probability, weight, outcome * weight))

    blocks = []
    for probability, weight, events in groups:
        blocks.append([probability, probability, weight, events / weight])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            merged_w = left[2] + right[2]
            merged_y = (left[2] * left[3] + right[2] * right[3]) / merged_w
            blocks.append([left[0], right[1], merged_w, merged_y])

    xs = [block[1] for block in blocks]
    ys = [block[3] for block in blocks]

    def interpolate(value):
        if value <= xs[0]:
            return ys[0]
        if value >= xs[-1]:
            return ys[-1]
        for index in range(len(xs) - 1):
            if xs[index] <= value <= xs[index + 1]:
                fraction = (value - xs[index]) / (xs[index + 1] - xs[index])
                return ys[index] + fraction * (ys[index + 1] - ys[index])
        raise AssertionError("interpolation failed")

    return [interpolate(value) for value in eval_p]
