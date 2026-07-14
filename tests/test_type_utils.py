"""Tests for type classification and compatibility."""

from typing import Literal

from outlines.types import CFG, Regex, cfg, regex
from pydantic import BaseModel

from outlines_cascade.type_utils import (
    OutputTypeCategory,
    classify_output_type,
    convert_to_json_compatible,
    entry_supports_category,
    is_cloud_compatible,
    is_local_compatible,
)

# ── test models ─────────────────────────────────────────────────────────


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float


class SimpleUser(BaseModel):
    name: str
    age: int


# ── classify_output_type ────────────────────────────────────────────────


class TestClassifyOutputType:
    def test_pydantic_model_is_json(self):
        cat, term = classify_output_type(Sentiment)
        assert cat == OutputTypeCategory.JSON

    def test_dict_is_json(self):
        cat, term = classify_output_type({"type": "object"})
        assert cat == OutputTypeCategory.JSON

    def test_literal_strings_is_choice(self):
        cat, term = classify_output_type(Literal["yes", "no", "maybe"])
        assert cat == OutputTypeCategory.CHOICE

    def test_regex_type_is_regex(self):
        cat, term = classify_output_type(regex(r"\d{4}"))
        assert cat == OutputTypeCategory.REGEX

    def test_raw_regex_pattern_is_regex(self):
        cat, term = classify_output_type(Regex(r"[0-9]+"))
        assert cat == OutputTypeCategory.REGEX

    def test_cfg_type_is_cfg(self):
        cat, term = classify_output_type(cfg("root := 'hello'"))
        assert cat == OutputTypeCategory.CFG

    def test_raw_cfg_is_cfg(self):
        cat, term = classify_output_type(CFG("root := 'hello'"))
        assert cat == OutputTypeCategory.CFG

    def test_int_is_regex(self):
        cat, term = classify_output_type(int)
        # int -> types.integer which is a Regex
        assert cat == OutputTypeCategory.REGEX

    def test_str_is_regex(self):
        cat, term = classify_output_type(str)
        # str -> types.string which is a Regex
        assert cat == OutputTypeCategory.REGEX


# ── compatibility checks ────────────────────────────────────────────────


class TestCompatibility:
    def test_json_cloud_compatible(self):
        assert is_cloud_compatible(OutputTypeCategory.JSON)

    def test_choice_cloud_compatible(self):
        assert is_cloud_compatible(OutputTypeCategory.CHOICE)

    def test_regex_not_cloud_compatible(self):
        assert not is_cloud_compatible(OutputTypeCategory.REGEX)

    def test_cfg_not_cloud_compatible(self):
        assert not is_cloud_compatible(OutputTypeCategory.CFG)

    def test_all_local_compatible(self):
        for cat in OutputTypeCategory:
            assert is_local_compatible(cat)

    def test_steerable_supports_all(self):
        """Steerable (local) models support everything."""
        for cat in OutputTypeCategory:
            assert entry_supports_category(cat, None, is_steerable=True)

    def test_cloud_auto_detect_json(self):
        assert entry_supports_category(
            OutputTypeCategory.JSON, None, is_steerable=False
        )

    def test_cloud_auto_detect_regex_rejected(self):
        assert not entry_supports_category(
            OutputTypeCategory.REGEX, None, is_steerable=False
        )

    def test_cloud_auto_detect_cfg_rejected(self):
        assert not entry_supports_category(
            OutputTypeCategory.CFG, None, is_steerable=False
        )

    def test_explicit_supported_types_override(self):
        """User can override auto-detection with explicit supported_types."""
        # Even a cloud entry can be marked as supporting regex
        assert entry_supports_category(
            OutputTypeCategory.REGEX,
            ["json", "regex"],
            is_steerable=False,
        )
        # And a steerable entry can be restricted
        assert not entry_supports_category(
            OutputTypeCategory.CFG,
            ["json"],
            is_steerable=True,
        )

    def test_explicit_override_case_insensitive(self):
        assert entry_supports_category(
            OutputTypeCategory.JSON,
            ["JSON", "Regex"],
            is_steerable=False,
        )


# ── convert_to_json_compatible ──────────────────────────────────────────


class TestConvertToJson:
    def test_choice_converts_to_literal(self):
        from outlines.types.dsl import Choice

        term = Choice(["yes", "no", "maybe"])
        result = convert_to_json_compatible(term)
        assert result is not None
        # Should be a typing.Literal
        import typing

        assert typing.get_args(result) == ("yes", "no", "maybe")

    def test_json_schema_passes_through(self):
        from outlines.types import json_schema

        term = json_schema({"type": "object", "properties": {}})
        result = convert_to_json_compatible(term)
        assert result is not None
        assert isinstance(result, dict)

    def test_regex_returns_none(self):
        from outlines.types import regex

        term = regex(r"\d+")
        result = convert_to_json_compatible(term)
        assert result is None
