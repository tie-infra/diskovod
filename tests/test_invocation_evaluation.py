from io import StringIO

import pytest

from diskovod.invocation_evaluation import (
    ALIAS_SENTINEL,
    EvaluationExample,
    evaluate,
    normalize_alias_candidate,
    read_jsonl,
)


def test_alias_sentinel_input_is_independent_of_the_configured_name():
    first = normalize_alias_candidate("thoughts, Diskovod", 10, 18)
    second = normalize_alias_candidate("thoughts, Дисковод", 10, 18)
    assert first == second == f"thoughts, {ALIAS_SENTINEL}"


def test_evaluation_reports_precision_first_quality_by_locale_and_latency():
    examples = (
        EvaluationExample("en", "Diskovod?", 0, 8, True, 0.99, 2),
        EvaluationExample("en", "I use Diskovod", 6, 14, False, 0.10, 4),
        EvaluationExample("fr", "Diskovod ?", 0, 8, True, 0.80, 6),
        EvaluationExample("fr", "j'aime Diskovod", 7, 15, False, 0.96, 8),
    )
    report = evaluate(examples, threshold=0.95)
    assert report.overall.count == 4
    assert report.overall.true_positive == 1
    assert report.overall.false_positive == 1
    assert report.overall.false_negative == 1
    assert report.locales["en"].precision == 1
    assert report.locales["en"].recall == 1
    assert report.locales["fr"].precision == 0
    assert report.latency_p50_ms == 5
    assert report.latency_p95_ms == pytest.approx(7.7)


def test_jsonl_reader_reports_the_failing_line():
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(
            StringIO(
                '{"locale":"en","text":"Diskovod","alias_start":0,'
                '"alias_end":8,"addressed":true,"score":0.9}\nnot json\n'
            )
        )
