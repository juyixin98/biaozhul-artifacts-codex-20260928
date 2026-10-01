import numpy as np

from core.binning import assign_bins, fixed_edges, quantile_edges


def test_fixed_boundaries_include_explicit_empty_bins():
    edges = fixed_edges(10)

    assert edges.tolist() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_tied_probabilities_are_not_split_by_equal_frequency_binning():
    p = np.array([0.2, 0.2, 0.8, 0.8])
    w = np.ones(4)

    edges = quantile_edges(p, w, 2)
    assignments = assign_bins(p, edges)

    assert edges.tolist() == [0.0, 0.5, 1.0]
    assert assignments.tolist() == [0, 0, 1, 1]


def test_quantile_cut_uses_gap_and_keeps_tie_with_target_weight():
    p = np.array([0.1, 0.2, 0.2, 0.9])
    w = np.array([1.0, 1.0, 1.0, 1.0])

    edges = quantile_edges(p, w, 2)
    assignments = assign_bins(p, edges)

    # Target mass is 2; the score 0.2 group has cumulative mass 3, so it stays left.
    assert edges.tolist() == [0.0, 0.55, 1.0]
    assert assignments.tolist() == [0, 0, 0, 1]


def test_too_many_requested_quantile_bins_become_visible_empty_slots():
    p = np.array([0.2, 0.8])
    w = np.ones(2)

    edges = quantile_edges(p, w, 4)
    assignments = assign_bins(p, edges)

    assert len(edges) == 5
    assert set(assignments.tolist()) == {0, 1}


def test_intervals_are_left_closed_and_final_bin_contains_one():
    edges = np.array([0.0, 0.1, 0.2, 1.0])

    assert assign_bins(np.array([0.0, 0.1, 0.19, 0.2, 1.0]), edges).tolist() == [0, 1, 1, 2, 2]


def test_zero_weight_rows_do_not_create_quantile_mass():
    p = np.array([0.2, 0.8, 0.9])
    w = np.array([1.0, 0.0, 1.0])

    edges = quantile_edges(p, w, 2)
    assignments = assign_bins(p, edges)

    assert edges.tolist() == [0.0, 0.5, 1.0]
    assert assignments.tolist() == [0, 1, 1]
