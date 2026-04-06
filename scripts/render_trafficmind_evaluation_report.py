"""Render a standalone TrafficMind evaluation HTML report from local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from trafficmind.evaluation import load_evaluation_artifacts, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a TrafficMind evaluation/admin report from local JSON artifacts.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Artifact JSON files or directories containing artifact JSON files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output HTML report.",
    )
    parser.add_argument(
        "--title",
        default="TrafficMind Evaluation Report",
        help="Page title shown in the report.",
    )
    args = parser.parse_args()

    artifacts = load_evaluation_artifacts(args.inputs)
    output_path = write_evaluation_report(
        artifacts,
        Path(args.output),
        title=args.title,
    )
    print(f"Wrote TrafficMind evaluation report to: {output_path}")


if __name__ == "__main__":
    main()