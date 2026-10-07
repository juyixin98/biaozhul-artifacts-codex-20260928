"""Small local algorithm package used by the mutation fixtures."""

from .pricing import discounted_total
from .search import find_index
from .sentinel import boot_sentinel
from .shapes import classify_size

__all__ = ["boot_sentinel", "classify_size", "discounted_total", "find_index"]
