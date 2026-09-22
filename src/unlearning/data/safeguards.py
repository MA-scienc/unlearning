from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class DatasetSeparationError(ValueError):
    """Raised when forget-set synthetic identifiers leak into specialization data."""


@dataclass(frozen=True)
class SensitiveTerms:
    terms: tuple[str, ...]


def collect_forget_sensitive_terms(records: Iterable[dict]) -> SensitiveTerms:
    terms: list[str] = []
    for record in records:
        for key in (
            "record_id",
            "patient_name",
            "synthetic_location",
            "clinic_id",
            "accession_id",
            "private_code",
        ):
            value = str(record.get(key) or "").strip()
            if value:
                terms.append(value)
    unique = tuple(sorted(set(terms), key=lambda item: item.lower()))
    return SensitiveTerms(unique)


def assert_no_forget_terms_in_texts(records: Iterable[dict], texts: Iterable[str]) -> None:
    terms = collect_forget_sensitive_terms(records).terms
    lower_terms = [(term, term.lower()) for term in terms if len(term) >= 4]
    for text_index, text in enumerate(texts):
        haystack = text.lower()
        for original, lowered in lower_terms:
            if lowered in haystack:
                raise DatasetSeparationError(
                    f"Forget-set synthetic term {original!r} appears in specialization text "
                    f"at index {text_index}"
                )
