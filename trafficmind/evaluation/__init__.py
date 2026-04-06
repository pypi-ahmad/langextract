"""Artifact-backed evaluation and benchmark report helpers for TrafficMind.

This package intentionally renders local evaluation artifacts or stored result
exports. It does not compute benchmark numbers on its own and does not claim to
be a live experiment-management system.
"""

from trafficmind.evaluation.io import load_evaluation_artifacts
from trafficmind.evaluation.models import EvaluationArtifact
from trafficmind.evaluation.models import EvaluationSample
from trafficmind.evaluation.models import EvaluationSection
from trafficmind.evaluation.models import EvaluationSectionKind
from trafficmind.evaluation.models import ManualReviewSummary
from trafficmind.evaluation.models import MeasuredMetric
from trafficmind.evaluation.models import PlaceholderNotice
from trafficmind.evaluation.models import RegistryBinding
from trafficmind.evaluation.models import RuleValidationScenario
from trafficmind.evaluation.report import render_evaluation_report
from trafficmind.evaluation.report import write_evaluation_report

__all__ = [
    "EvaluationArtifact",
    "EvaluationSample",
    "EvaluationSection",
    "EvaluationSectionKind",
    "ManualReviewSummary",
    "MeasuredMetric",
    "PlaceholderNotice",
    "RegistryBinding",
    "RuleValidationScenario",
    "load_evaluation_artifacts",
    "render_evaluation_report",
    "write_evaluation_report",
]
