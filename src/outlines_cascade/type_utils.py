"""Output-type classification and compatibility checking.

Bridges the gap between Outlines' rich type system and the reality that
cloud API providers only support JSON Schema, while local models support
all types (regex, CFG, etc.) via FSM-based constrained decoding.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from outlines.types.dsl import (
    CFG,
    Alternatives,
    Choice,
    JsonSchema,
    Regex,
    Term,
    python_types_to_terms,
)


class OutputTypeCategory(str, Enum):
    """High-level category of an output type.

    Members:
        JSON:   Pydantic models, dataclasses, typed dicts, JSON Schema dicts.
                Supported by both cloud and local models.
        REGEX:  Regex-constrained text.  Only supported by local models.
        CFG:    Context-free grammar.  Only supported by local models.
        CHOICE: Multiple-choice selection.  Convertible to JSON for cloud.
        OTHER:  Any Term not classified above (e.g. raw String).
    """

    JSON = "json"
    REGEX = "regex"
    CFG = "cfg"
    CHOICE = "choice"
    OTHER = "other"


# ── cloud-capable categories ────────────────────────────────────────────

CLOUD_SUPPORTED: frozenset[OutputTypeCategory] = frozenset({
    OutputTypeCategory.JSON,
    OutputTypeCategory.CHOICE,
})

ALL_SUPPORTED: frozenset[OutputTypeCategory] = frozenset({
    OutputTypeCategory.JSON,
    OutputTypeCategory.REGEX,
    OutputTypeCategory.CHOICE,
    OutputTypeCategory.CFG,
    OutputTypeCategory.OTHER,
})


def classify_output_type(output_type: Any) -> tuple[OutputTypeCategory, Term]:
    """Classify a user-provided output type into a category.

    Uses Outlines' own ``python_types_to_terms`` to normalise the input,
    then inspects the resulting :class:`Term` to determine the category.

    Parameters
    ----------
    output_type
        The user-provided output type: Pydantic model, dataclass, typed
        dict, JSON Schema dict, ``Literal[...]``, :class:`outlines.types.Regex`,
        :class:`outlines.types.CFG`, :class:`outlines.types.Choice`, etc.

    Returns
    -------
    tuple[OutputTypeCategory, Term]
        The category and the normalised Outlines Term.
    """
    # Raw dicts that look like JSON Schemas are handled specially —
    # python_types_to_terms treats them as "native dict" (CFG(json grammar)),
    # but the user almost certainly means a JSON Schema.
    if isinstance(output_type, dict):
        from outlines.types import JsonSchema as JsonSchemaType

        term = JsonSchemaType(output_type)
        return OutputTypeCategory.JSON, term

    term = python_types_to_terms(output_type)

    if isinstance(term, JsonSchema):
        return OutputTypeCategory.JSON, term

    if isinstance(term, CFG):
        return OutputTypeCategory.CFG, term

    if isinstance(term, Choice):
        return OutputTypeCategory.CHOICE, term

    if isinstance(term, Regex):
        return OutputTypeCategory.REGEX, term

    if isinstance(term, Alternatives):
        # Alternatives can be Literal["a","b"] (like choice) or a union
        # of complex types.  Check whether all sub-terms are simple
        # strings/literals — if so, treat as CHOICE (cloud-compatible).
        if _all_simple_alternatives(term):
            return OutputTypeCategory.CHOICE, term
        # Otherwise, fall back to regex (the universal local-model path).
        return OutputTypeCategory.REGEX, term

    return OutputTypeCategory.OTHER, term


_REGEX_METACHARS = set(".^$*+?{}[]|\\")


def _all_simple_alternatives(term: Alternatives) -> bool:
    """Check whether every sub-term in an Alternatives is a simple literal."""
    from outlines.types.dsl import String

    for sub in term.terms:
        if not isinstance(sub, (String, Regex)):
            return False
        # Regex is OK only if it's a simple escaped literal
        if isinstance(sub, Regex) and (set(sub.pattern) & _REGEX_METACHARS):
            return False
    return True


def is_cloud_compatible(category: OutputTypeCategory) -> bool:
    """Return True if cloud API providers can handle this category."""
    return category in CLOUD_SUPPORTED


def is_local_compatible(category: OutputTypeCategory) -> bool:
    """Return True if local (steerable) models can handle this category."""
    return category in ALL_SUPPORTED


def entry_supports_category(
    category: OutputTypeCategory,
    supported_types: list[str] | None,
    is_steerable: bool,
) -> bool:
    """Check whether a cascade entry supports a given output type category.

    Resolution logic:
    1. If the entry has an explicit ``supported_types`` override, use it.
    2. Otherwise, auto-detect: steerable models (local) support everything;
       black-box models (cloud) support JSON + Choice.

    Parameters
    ----------
    category
        The output type category to check.
    supported_types
        Explicit override from config (list of category strings).
        If ``None``, auto-detection is used.
    is_steerable
        Whether the entry's provider is a steerable (local) model.

    Returns
    -------
    bool
        Whether the entry can handle this category.
    """
    if supported_types is not None:
        explicit = frozenset(s.lower() for s in supported_types)
        return category.value in explicit

    if is_steerable:
        return True
    return is_cloud_compatible(category)


def convert_to_json_compatible(term: Term) -> type | dict | None:
    """Attempt to convert an Outlines Term to a JSON-Schema-compatible type.

    This is used when a CHOICE type needs to be routed to a cloud provider.

    Returns the converted type, or ``None`` if conversion is not possible.
    """
    from outlines.types.dsl import Choice

    if isinstance(term, JsonSchema):
        return JsonSchema.convert_to(term, ["dict"])

    if isinstance(term, Choice):
        # Convert choice items to a Pydantic model with a single field
        from typing import Literal

        items = term.items
        # Only works if all items are strings
        if all(isinstance(i, str) for i in items):
            return Literal[tuple(items)]  # type: ignore[return-value]
        return None

    if isinstance(term, Alternatives):
        from typing import Literal

        from outlines.types.dsl import String

        # If all alternatives are simple strings, build a Literal
        literal_values = []
        for sub in term.terms:
            if isinstance(sub, String):
                literal_values.append(sub.value)
            elif isinstance(sub, Regex):
                # Check it's a simple literal
                if set(sub.pattern) & _REGEX_METACHARS:
                    return None
                literal_values.append(sub.pattern)
            else:
                return None
        if literal_values:
            return Literal[tuple(literal_values)]  # type: ignore[return-value]
        return None

    return None
