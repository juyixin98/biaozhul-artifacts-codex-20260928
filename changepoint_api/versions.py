"""Version identifiers persisted with each result."""

from . import __version__

PACKAGE_VERSION = __version__
ALGORITHM_VERSION = "offline-mean-segmentation-v1"
KERNEL_VERSION = "longdouble-neumaier-prefix-dp-v1"
PRUNED_KERNEL_VERSION = "finite-horizon-dominance-pruned-dp-v1"
TIE_BREAK_RULE = (
    "objective ascending, then more segments at equal cost, then smaller "
    "final change-point; predecessor uniquely fixes earlier boundaries"
)
