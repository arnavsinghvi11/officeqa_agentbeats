import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from agent import (
    normalize_text,
    extract_numbers_with_context,
    detect_unit_in_context,
    normalize_number_with_units,
    is_likely_year,
    has_significant_text,
    check_text_overlap,
    extract_final_answer,
    contains_multiple_candidates,
    fuzzy_match_answer,
    score_answer,
)


class TestNormalizeText:
    def test_unicode_minus_normalization(self):
        assert normalize_text("100\u2212200") == "100-200"
        assert normalize_text("50−30") == "50-30"

    def test_regular_text_unchanged(self):
        assert normalize_text("hello world") == "hello world"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            normalize_text("")


class TestExtractNumbersWithContext:
    def test_simple_integer(self):
        results = extract_numbers_with_context("The value is 2602")
        assert len(results) == 1
        assert results[0][0] == 2602.0

    def test_comma_separated_number(self):
        results = extract_numbers_with_context("Total: 2,602 million")
        assert len(results) == 1
        assert results[0][0] == 2602.0

    def test_percentage(self):
        results = extract_numbers_with_context("Growth rate: 15.3%")
        assert len(results) == 1
        assert results[0][0] == 15.3
        assert results[0][2] is True

    def test_negative_number(self):
        results = extract_numbers_with_context("Deficit: -500 million")
        assert len(results) == 1
        assert results[0][0] == -500.0
        assert results[0][3] is True

    def test_multiple_numbers(self):
        results = extract_numbers_with_context("From 100 to 200")
        assert len(results) == 2
        assert results[0][0] == 100.0
        assert results[1][0] == 200.0

    def test_decimal_number(self):
        results = extract_numbers_with_context("Rate is 3.75")
        assert len(results) == 1
        assert results[0][0] == 3.75

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            extract_numbers_with_context("")


class TestDetectUnitInContext:
    def test_trillion(self):
        unit, mult = detect_unit_in_context("5 trillion dollars")
        assert unit == "trillion"
        assert mult == 1e12

    def test_billion(self):
        unit, mult = detect_unit_in_context("2.5 billion")
        assert unit == "billion"
        assert mult == 1e9

    def test_million(self):
        unit, mult = detect_unit_in_context("100 million")
        assert unit == "million"
        assert mult == 1e6

    def test_thousand(self):
        unit, mult = detect_unit_in_context("50 thousand")
        assert unit == "thousand"
        assert mult == 1e3

    def test_no_unit(self):
        unit, mult = detect_unit_in_context("just 500")
        assert unit is None
        assert mult == 1.0

    def test_abbreviation_b_word_boundary(self):
        unit, mult = detect_unit_in_context("$5 b total")
        assert unit == "billion"

    def test_abbreviation_m_word_boundary(self):
        unit, mult = detect_unit_in_context("$100 m spent")
        assert unit == "million"


class TestIsLikelyYear:
    def test_valid_year(self):
        assert is_likely_year(1940) is True
        assert is_likely_year(2025) is True
        assert is_likely_year(1939) is True

    def test_invalid_year_too_old(self):
        assert is_likely_year(1899) is False

    def test_invalid_year_too_new(self):
        assert is_likely_year(2101) is False

    def test_non_integer(self):
        assert is_likely_year(1940.5) is False

    def test_regular_number(self):
        assert is_likely_year(500) is False
        assert is_likely_year(2602) is False


class TestHasSignificantText:
    def test_purely_numeric(self):
        has_text, cleaned = has_significant_text("2,602")
        assert has_text is False

    def test_with_unit_words_only(self):
        has_text, cleaned = has_significant_text("100 million")
        assert has_text is False

    def test_with_significant_text(self):
        has_text, cleaned = has_significant_text("January 1945")
        assert has_text is True
        assert "january" in cleaned

    def test_text_with_numbers(self):
        has_text, cleaned = has_significant_text("Federal Reserve 2020")
        assert has_text is True


class TestCheckTextOverlap:
    def test_purely_numeric_gt(self):
        matches, rationale = check_text_overlap("2602", "The answer is 2602")
        assert matches is True

    def test_text_overlap_found(self):
        matches, rationale = check_text_overlap("January 1945", "The date was January 1945")
        assert matches is True

    def test_text_mismatch(self):
        matches, rationale = check_text_overlap("January 1945", "February 1946")
        assert matches is False

    def test_gt_has_text_pred_numeric(self):
        matches, rationale = check_text_overlap("January 1945", "1945")
        assert matches is False


class TestExtractFinalAnswer:
    def test_valid_extraction(self):
        text = "<REASONING>Some steps</REASONING><FINAL_ANSWER>2602</FINAL_ANSWER>"
        assert extract_final_answer(text) == "2602"

    def test_with_whitespace(self):
        text = "<FINAL_ANSWER>  2602  </FINAL_ANSWER>"
        assert extract_final_answer(text) == "2602"

    def test_case_insensitive(self):
        text = "<final_answer>2602</final_answer>"
        assert extract_final_answer(text) == "2602"

    def test_missing_tags_raises(self):
        with pytest.raises(ValueError, match="No FINAL_ANSWER tags found"):
            extract_final_answer("Just some text without tags")

    def test_empty_tags_raises(self):
        with pytest.raises(ValueError, match="FINAL_ANSWER tags are empty"):
            extract_final_answer("<FINAL_ANSWER></FINAL_ANSWER>")

    def test_too_long_raises(self):
        long_answer = "x" * 501
        with pytest.raises(ValueError, match="too long"):
            extract_final_answer(f"<FINAL_ANSWER>{long_answer}</FINAL_ANSWER>")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            extract_final_answer("")


class TestContainsMultipleCandidates:
    def test_single_value_no_hedge(self):
        is_hedged, reason = contains_multiple_candidates("2602", "The answer is 2602")
        assert is_hedged is False

    def test_multiple_candidates_hedged(self):
        is_hedged, reason = contains_multiple_candidates("2602", "Either 2602 or 2700 or 2800")
        assert is_hedged is True
        assert "3 candidates" in reason

    def test_gt_has_multiple_values(self):
        is_hedged, reason = contains_multiple_candidates("100, 200, 300", "100, 200, 300")
        assert is_hedged is False

    def test_year_not_counted_as_hedge(self):
        is_hedged, reason = contains_multiple_candidates("2602", "In 1940, the value was 2602")
        assert is_hedged is False


class TestFuzzyMatchAnswer:
    def test_exact_numeric_match(self):
        matches, rationale = fuzzy_match_answer("2602", "2602")
        assert matches is True

    def test_comma_formatting_match(self):
        matches, rationale = fuzzy_match_answer("2,602", "2602")
        assert matches is True

    def test_within_tolerance(self):
        matches, rationale = fuzzy_match_answer("100", "99", tolerance=0.05)
        assert matches is True

    def test_outside_tolerance(self):
        matches, rationale = fuzzy_match_answer("100", "90", tolerance=0.05)
        assert matches is False

    def test_zero_tolerance_exact(self):
        matches, rationale = fuzzy_match_answer("2602", "2602", tolerance=0.0)
        assert matches is True

    def test_zero_tolerance_mismatch(self):
        matches, rationale = fuzzy_match_answer("2602", "2603", tolerance=0.0)
        assert matches is False

    def test_text_match(self):
        matches, rationale = fuzzy_match_answer("January", "January")
        assert matches is True

    def test_text_in_prediction(self):
        matches, rationale = fuzzy_match_answer("January", "The month was January")
        assert matches is True

    def test_unit_aware_matching(self):
        matches, rationale = fuzzy_match_answer("2.6 billion", "2.6 billion")
        assert matches is True

    def test_percentage_match(self):
        matches, rationale = fuzzy_match_answer("15.3%", "15.3%")
        assert matches is True

    def test_hedged_answer_rejected(self):
        matches, rationale = fuzzy_match_answer("100", "Either 100 or 200 or 300")
        assert matches is False
        assert "Hedged" in rationale

    def test_year_filtered_from_comparison(self):
        matches, rationale = fuzzy_match_answer("500", "In 1940, the value was 500")
        assert matches is True

    def test_multiple_gt_values_all_match(self):
        matches, rationale = fuzzy_match_answer("100, 200", "100 and 200")
        assert matches is True

    def test_multiple_gt_values_partial_match(self):
        matches, rationale = fuzzy_match_answer("100, 200, 300", "100 and 200")
        assert matches is False


class TestScoreAnswer:
    def test_correct_with_tags(self):
        gt = "2602"
        pred = "<REASONING>Calculated from table</REASONING><FINAL_ANSWER>2602</FINAL_ANSWER>"
        is_correct, rationale = score_answer(gt, pred)
        assert is_correct is True

    def test_incorrect_value(self):
        gt = "2602"
        pred = "<FINAL_ANSWER>1700</FINAL_ANSWER>"
        is_correct, rationale = score_answer(gt, pred, tolerance=0.0)
        assert is_correct is False

    def test_missing_tags(self):
        gt = "2602"
        pred = "The answer is 2602"
        is_correct, rationale = score_answer(gt, pred)
        assert is_correct is False
        assert "No FINAL_ANSWER tags found" in rationale

    def test_no_answer_found_response(self):
        gt = "2602"
        pred = "<FINAL_ANSWER>no answer found</FINAL_ANSWER>"
        is_correct, rationale = score_answer(gt, pred)
        assert is_correct is False
        assert "no answer found" in rationale

    def test_with_tolerance(self):
        gt = "100"
        pred = "<FINAL_ANSWER>99</FINAL_ANSWER>"
        is_correct, rationale = score_answer(gt, pred, tolerance=0.05)
        assert is_correct is True

    def test_strict_tolerance(self):
        gt = "100"
        pred = "<FINAL_ANSWER>99</FINAL_ANSWER>"
        is_correct, rationale = score_answer(gt, pred, tolerance=0.0)
        assert is_correct is False


class TestEdgeCases:
    def test_negative_value_matching(self):
        matches, rationale = fuzzy_match_answer("-500", "-500")
        assert matches is True

    def test_zero_value_matching(self):
        matches, rationale = fuzzy_match_answer("0", "0")
        assert matches is True

    def test_large_number_matching(self):
        matches, rationale = fuzzy_match_answer("1000000000", "1000000000")
        assert matches is True

    def test_decimal_precision(self):
        matches, rationale = fuzzy_match_answer("3.14159", "3.14159")
        assert matches is True

    def test_unicode_minus_in_answer(self):
        gt = "100"
        pred = "<FINAL_ANSWER>100</FINAL_ANSWER>"
        is_correct, _ = score_answer(gt, pred)
        assert is_correct is True
