"""Custom exceptions for TrafficMind signal integration."""

from __future__ import annotations


class TrafficMindError(Exception):
  """Base exception for all TrafficMind errors."""


class SourceUnavailableError(TrafficMindError):
  """A signal source could not be reached or is not configured."""


class StaleSignalError(TrafficMindError):
  """A signal observation is older than the allowed staleness threshold."""


class SignalConflictError(TrafficMindError):
  """Controller and vision disagree and cannot be automatically resolved."""


class InvalidSignalDataError(TrafficMindError):
  """Inbound signal data failed validation."""
