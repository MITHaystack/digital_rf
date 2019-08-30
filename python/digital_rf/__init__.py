"""Digital RF Python package."""

from .digital_metadata import *
from .digital_rf_hdf5 import *
from . import list_drf
from .list_drf import ilsdrf, lsdrf
from . import util

try:
    from . import mirror
    from . import ringbuffer
    from . import watchdog_drf
except ImportError:
    # if no watchdog package, these fail to import, so just ignore
    pass

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
