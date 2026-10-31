"""Detect and anonymize personally identifiable information with Presidio."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass(frozen=True)
class PIIFinding:
    """A PII match without copying the matched sensitive value."""

    entity_type: str
    start: int

    end: int
    score: float


@dataclass(frozen=True)
class PIIAnonymizationResult:
    """Anonymized text and the locations/types of the matches in the input."""

    text: str
    findings: tuple[PIIFinding, ...]


class PIIHandler:
    """Detect and anonymize PII in text using Microsoft Presidio.

    Presidio's default English NLP engine requires the spaCy model
    ``en_core_web_lg``. Install it with ``python -m spacy download en_core_web_lg``
    after installing this project's requirements.
    """

    def __init__(
        self,
        language: str = "en",
        analyzer: AnalyzerEngine | None = None,
        anonymizer: AnonymizerEngine | None = None,
    ) -> None:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")

        self.language = language
        self._analyzer = analyzer if analyzer is not None else AnalyzerEngine()
        self._anonymizer = anonymizer if anonymizer is not None else AnonymizerEngine()

    def detect(
        self,
        text: str,
        entities: Sequence[str] | None = None,
        score_threshold: float = 0.5,
    ) -> list[PIIFinding]:
        """Return PII entity types, character spans, and confidence scores."""
        results = self._analyze(text, entities, score_threshold)
        return [
            PIIFinding(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
            )
            for result in results
        ]

    def anonymize(
        self,
        text: str,
        entities: Sequence[str] | None = None,
        score_threshold: float = 0.5,
        operators: Mapping[str, OperatorConfig] | None = None,
    ) -> PIIAnonymizationResult:
        """Replace detected PII with ``<PII>`` unless operators are supplied.

        Operators can be configured globally with the ``DEFAULT`` key or for
        individual Presidio entity types, such as ``EMAIL_ADDRESS``.
        """
        results = self._analyze(text, entities, score_threshold)
        configured_operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})
        }
        if operators is not None:
            configured_operators.update(operators)

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=configured_operators,
        )
        findings = tuple(
            PIIFinding(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
            )
            for result in results
        )
        return PIIAnonymizationResult(text=anonymized.text, findings=findings)

    def _analyze(
        self,
        text: str,
        entities: Sequence[str] | None,
        score_threshold: float,
    ) -> list[RecognizerResult]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(score_threshold, (int, float)) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if entities is not None and (
            isinstance(entities, str)
            or not all(isinstance(entity, str) and entity for entity in entities)
        ):
            raise ValueError("entities must be a sequence of non-empty strings")

        return self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=list(entities) if entities is not None else None,
            score_threshold=score_threshold,
        )