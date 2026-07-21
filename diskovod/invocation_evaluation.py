from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


ALIAS_SENTINEL = "<ASSISTANT_NAME>"


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    locale: str
    text: str
    alias_start: int
    alias_end: int
    addressed: bool
    score: float
    latency_ms: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EvaluationExample:
        example = cls(
            locale=str(value["locale"]),
            text=str(value["text"]),
            alias_start=int(value["alias_start"]),
            alias_end=int(value["alias_end"]),
            addressed=bool(value["addressed"]),
            score=float(value["score"]),
            latency_ms=(float(value["latency_ms"]) if value.get("latency_ms") is not None else None),
        )
        if not 0 <= example.alias_start < example.alias_end <= len(example.text):
            raise ValueError("alias span is outside the example text")
        if not math.isfinite(example.score) or not 0 <= example.score <= 1:
            raise ValueError("score must be finite and between zero and one")
        if example.latency_ms is not None and (
            not math.isfinite(example.latency_ms) or example.latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        return example

    def classifier_text(self) -> str:
        return normalize_alias_candidate(self.text, self.alias_start, self.alias_end)


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    threshold: float
    overall: QualityMetrics
    locales: dict[str, QualityMetrics]
    latency_samples: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_alias_candidate(text: str, alias_start: int, alias_end: int) -> str:
    if not 0 <= alias_start < alias_end <= len(text):
        raise ValueError("alias span is outside the candidate text")
    return text[:alias_start] + ALIAS_SENTINEL + text[alias_end:]


def evaluate(examples: Iterable[EvaluationExample], *, threshold: float) -> EvaluationReport:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    materialized = tuple(examples)
    if not materialized:
        raise ValueError("the evaluation dataset is empty")
    overall = _metrics(materialized, threshold)
    locales = {
        locale: _metrics(tuple(item for item in materialized if item.locale == locale), threshold)
        for locale in sorted({item.locale for item in materialized})
    }
    latencies = sorted(item.latency_ms for item in materialized if item.latency_ms is not None)
    return EvaluationReport(
        threshold=threshold,
        overall=overall,
        locales=locales,
        latency_samples=len(latencies),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def read_jsonl(stream: TextIO) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record must be a JSON object")
            examples.append(EvaluationExample.from_dict(value))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSONL record on line {line_number}: {error}") from error
    return tuple(examples)


def _metrics(examples: tuple[EvaluationExample, ...], threshold: float) -> QualityMetrics:
    true_positive = false_positive = true_negative = false_negative = 0
    for example in examples:
        prediction = example.score >= threshold
        if prediction and example.addressed:
            true_positive += 1
        elif prediction:
            false_positive += 1
        elif example.addressed:
            false_negative += 1
        else:
            true_negative += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return QualityMetrics(
        count=len(examples),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate offline invocation-classifier scores without enabling runtime classification."
    )
    parser.add_argument("dataset", type=Path, help="JSONL dataset containing labels and candidate scores")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        with args.dataset.open(encoding="utf-8") as stream:
            report = evaluate(read_jsonl(stream), threshold=args.threshold)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
