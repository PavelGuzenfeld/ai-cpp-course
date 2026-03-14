"""tracker-utils: C++ accelerated tracking utilities."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tracker-utils")
except PackageNotFoundError:
    # Package is not installed (running from source without pip install)
    __version__ = "0.0.0-dev"

# Import the C++ native extension
try:
    from tracker_utils._native import BBox as _NativeBBox
except ImportError as e:
    raise ImportError(
        "Failed to import the C++ native extension. "
        "Make sure the package was built correctly with 'pip install .' "
        f"Original error: {e}"
    ) from e

# Re-export the high-level Python wrapper
from tracker_utils.bbox import BBox

__all__ = ["BBox", "__version__"]
