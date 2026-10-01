import numpy as np
import pytest

from app.core.scaling import ParameterScaling


def test_parameter_scaling_round_trip_and_chain_derivative():
    scaling = ParameterScaling.create(
        lower=np.array([0.0, -np.inf, -2.0]),
        upper=np.array([10.0, np.inf, np.inf]),
        scales=np.array([2.0, 0.5, 1.5]),
    )
    theta = np.array([1.5, 3.0, -0.25])
    z = scaling.to_internal(theta)

    np.testing.assert_allclose(scaling.to_external(z), theta, rtol=0.0, atol=1.0e-12)

    # Numeric dtheta/dz independently checks the explicit chain-rule factors.
    h = 1.0e-7
    for i in range(3):
        shifted = z.copy()
        shifted[i] += h
        numeric_derivative = (scaling.to_external(shifted)[i] - theta[i]) / h
        assert numeric_derivative == pytest.approx(
            scaling.external_gradient_from_internal(z)[i], rel=1.0e-6
        )
