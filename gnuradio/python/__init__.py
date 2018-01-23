"""Digital RF module for GNU Radio."""

try:
    from ._version import __version__
except ImportError:
    __version__ = None
from .digital_rf_source import digital_rf_source, digital_rf_channel_source
from .digital_rf_sink import digital_rf_sink, digital_rf_channel_sink
