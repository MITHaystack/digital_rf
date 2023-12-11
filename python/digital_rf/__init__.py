"""Digital RF Python package."""

import logging as _logging
import os as _os

from .digital_metadata import *  # noqa: F401,F403
from .digital_rf_hdf5 import *  # noqa: F401,F403
from . import list_drf  # noqa: F401
from .list_drf import ilsdrf, lsdrf  # noqa: F401
from . import util  # noqa: F401
from . import _version

_logging.basicConfig(
    level=_os.environ.get(
        "DRF_LOGLEVEL", _os.environ.get("LOGLEVEL", "WARNING")
    ).upper()
)
_logger = _logging.getLogger(__name__)

try:
    from . import mirror  # noqa: F401
    from . import ringbuffer  # noqa: F401
    from . import watchdog_drf  # noqa: F401
except ImportError:
    # if no watchdog package, these fail to import, so warn and continue
    watchdog_msg = (
        "Failed to import the `mirror`, `ringbuffer`, and/or `watchdog_drf` modules,"
        " likely because the `watchdog` package is not installed or is incompatible."
        " You can safely ignore this if you do not need the functionality of any of"
        " those modules."
    )
    _logger.info(watchdog_msg, exc_info=True)

__version__ = _version.get_versions()["version"]
