"""Independent reference math for test oracle values.

This module deliberately does NOT import sgns.kernel. It implements the
skip-gram negative sampling update as explicit scalar operations so the
kernel test compares two independently written computations, plus concrete
constants that were computed by hand for a fixed two-dimensional fixture.
"""

from __future__ import annotations

import math


def ref_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def ref_event(
    u_before: list[float],
    output_rows: dict[int, list[float]],
    positive_id: int,
    negative_ids: list[int],
    lr: float,
) -> tuple[list[float], list[float], list[float], dict[int, list[float]]]:
    """Independent scalar event update.

    Returns (scores, coefficients, u_after, v_after_by_target).
    Positive uses (1 - sigmoid), negatives use -sigmoid.
    """
    targets = [positive_id, *negative_ids]
    scores: list[float] = []
    coefficients: list[float] = []
    for rank, target_id in enumerate(targets):
        score = sum(ui * vi for ui, vi in zip(u_before, output_rows[target_id]))
        if rank == 0:
            coefficient = lr * (1.0 - ref_sigmoid(score))
        else:
            coefficient = -lr * ref_sigmoid(score)
        scores.append(score)
        coefficients.append(coefficient)

    grad_u = [0.0 for _ in u_before]
    row_delta: dict[int, list[float]] = {}
    for target_id, coefficient in zip(targets, coefficients):
        for j, ui in enumerate(u_before):
            grad_u[j] += coefficient * output_rows[target_id][j]
        row_delta.setdefault(target_id, [0.0 for _ in u_before])
        for j, ui in enumerate(u_before):
            row_delta[target_id][j] += coefficient * ui

    u_after = [ui + gi for ui, gi in zip(u_before, grad_u)]
    v_after: dict[int, list[float]] = {}
    for target_id, delta in row_delta.items():
        v_after[target_id] = [
            vi + di for vi, di in zip(output_rows[target_id], delta)
        ]
    return scores, coefficients, u_after, v_after


# --- Hand-computed constants ------------------------------------------------
# dim=2, lr=0.1:
#   u(c) = [0.2, -0.4], v(p) = [0.5, 0.1],
#   v(n1) = [-0.3, 0.2], v(n2) = [0.7, -0.6]
# scores:
#   pos  = 0.2*0.5 + (-0.4)*0.1 = 0.06
#   neg1 = 0.2*-0.3 + (-0.4)*0.2 = -0.14
#   neg2 = 0.2*0.7 + (-0.4)*-0.6 = 0.38
HAND_LR = 0.1
HAND_U = [0.2, -0.4]
HAND_VP = [0.5, 0.1]
HAND_VN1 = [-0.3, 0.2]
HAND_VN2 = [0.7, -0.6]
HAND_SCORES = [0.06, -0.14, 0.38]
# sigmoid values independently computed to 12 decimal places:
#   sig(0.06)=0.514995501619, sig(-0.14)=0.465057054842, sig(0.38)=0.593873102934
HAND_SIGMOIDS = [0.514995501619, 0.465057054842, 0.593873102934]
HAND_COEFFICIENTS = [
    HAND_LR * (1.0 - HAND_SIGMOIDS[0]),
    -HAND_LR * HAND_SIGMOIDS[1],
    -HAND_LR * HAND_SIGMOIDS[2],
]
# u_after components, accumulated as g_pos*vp + g1*vn1 + g2*vn2:
HAND_GRAD_U = [
    HAND_COEFFICIENTS[0] * 0.5
    + HAND_COEFFICIENTS[1] * -0.3
    + HAND_COEFFICIENTS[2] * 0.7,
    HAND_COEFFICIENTS[0] * 0.1
    + HAND_COEFFICIENTS[1] * 0.2
    + HAND_COEFFICIENTS[2] * -0.6,
]
HAND_U_AFTER = [0.2 + HAND_GRAD_U[0], -0.4 + HAND_GRAD_U[1]]
# Output rows move by their own coefficient times the pre-update u:
HAND_VP_AFTER = [
    0.5 + HAND_COEFFICIENTS[0] * 0.2,
    0.1 + HAND_COEFFICIENTS[0] * -0.4,
]
HAND_VN1_AFTER = [
    -0.3 + HAND_COEFFICIENTS[1] * 0.2,
    0.2 + HAND_COEFFICIENTS[1] * -0.4,
]
HAND_VN2_AFTER = [
    0.7 + HAND_COEFFICIENTS[2] * 0.2,
    -0.6 + HAND_COEFFICIENTS[2] * -0.4,
]
