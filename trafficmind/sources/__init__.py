"""Signal source adapters."""

from trafficmind.sources.base import SignalSource
from trafficmind.sources.file_feed import FileFeedSource
from trafficmind.sources.polling import PollingSource
from trafficmind.sources.simulator import SimulatorSource
from trafficmind.sources.webhook import WebhookReceiver

__all__ = [
    "FileFeedSource",
    "PollingSource",
    "SignalSource",
    "SimulatorSource",
    "WebhookReceiver",
]
