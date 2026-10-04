"""Automated Delimiter Discovery module for NeuroFuzz (Phase 7D).

Analyzes structured text and byte inputs to discover protocol field delimiters
at runtime using multi-signal structural evidence (corpus presence, frequency
distribution, field length regularity, and structural punctuation priors)
without requiring target-specific hard-coding or building full formal grammars.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


# Standard protocol framing delimiters and their structural punctuation priors
STANDARD_DELIMITER_PRIORS: Dict[bytes, float] = {
    b"|": 1.0,   # Pipe-delimited protocols (Syslog, HL7, NeuroFuzz)
    b";": 1.0,   # Semicolon-delimited records (CSV-variant, SIP, Cookie)
    b",": 1.0,   # Comma-separated values (CSV, RFC 4180)
    b":": 0.95,  # Colon-delimited headers (HTTP, YAML, key-value)
    b"\t": 0.90, # Tab-separated values (TSV)
    b"=": 0.70,  # Key-value assignment (URL query parameters, env)
    b"&": 0.70,  # Query parameter separator
    b"#": 0.65,  # Hash/comment or anchor separator
    b"/": 0.60,  # Path delimiter
    b"-": 0.50,  # Dash separator (often in compound identifiers)
    b"_": 0.45,  # Underscore separator (identifier naming)
    b".": 0.40,  # Dot notation (extensions, subfields)
    b" ": 0.40,  # Space separator (whitespace-delimited words)
}

# Control characters and whitespace to strictly blacklist as protocol delimiters
DELIMITER_BLACKLIST: Set[int] = {
    0x00,  # NUL
    0x0A,  # LF (\n)
    0x0D,  # CR (\r)
    0x1B,  # ESC
}


@dataclass
class DelimiterCandidate:
    """Represents an evaluated delimiter candidate with explainable structural evidence."""

    delimiter: bytes
    confidence: float
    presence_ratio: float
    frequency_per_seed: float
    field_regularity_score: float
    punctuation_prior: float
    avg_field_length: float
    distinct_fields_found: int
    selection_reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to a serializable dictionary."""
        return {
            "delimiter": self.delimiter.decode("latin1", errors="replace"),
            "delimiter_hex": self.delimiter.hex(),
            "confidence": round(self.confidence, 4),
            "presence_ratio": round(self.presence_ratio, 4),
            "frequency_per_seed": round(self.frequency_per_seed, 2),
            "field_regularity_score": round(self.field_regularity_score, 4),
            "punctuation_prior": round(self.punctuation_prior, 2),
            "avg_field_length": round(self.avg_field_length, 2),
            "distinct_fields_found": self.distinct_fields_found,
            "selection_reason": self.selection_reason,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


class DelimiterDetector:
    """Runtime heuristic delimiter discovery engine for structured text/byte payloads."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        min_occurrences_per_seed: float = 1.0,
        min_corpus_presence: float = 0.20,
        default_delimiter: bytes = b"|",
        candidate_bytes: Optional[Sequence[bytes]] = None,
    ) -> None:
        """Initialize the delimiter detector.

        Args:
            confidence_threshold: Minimum composite confidence score required to accept a delimiter.
            min_occurrences_per_seed: Minimum average occurrences per seed required.
            min_corpus_presence: Minimum fraction of corpus seeds containing the delimiter.
            default_delimiter: Fallback delimiter if auto-discovery produces low confidence.
            candidate_bytes: Optional list of candidate byte sequences to evaluate.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.min_occurrences_per_seed = float(min_occurrences_per_seed)
        self.min_corpus_presence = float(min_corpus_presence)
        self.default_delimiter = bytes(default_delimiter)

        if candidate_bytes is not None:
            self.candidate_bytes = [bytes(c) for c in candidate_bytes if len(c) > 0]
        else:
            # Standard candidate evaluation set: framing delimiters + printable non-alphanumeric ASCII
            candidates = list(STANDARD_DELIMITER_PRIORS.keys())
            for b in range(0x21, 0x7F):
                char_byte = bytes([b])
                # Skip alphanumeric ASCII and existing candidates
                if not (0x30 <= b <= 0x39 or 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A):
                    if char_byte not in candidates and b not in DELIMITER_BLACKLIST:
                        candidates.append(char_byte)
            self.candidate_bytes = candidates

        self._cached_results: Optional[List[DelimiterCandidate]] = None
        self._cached_corpus_hash: Optional[str] = None

    def score_candidate(
        self,
        candidate: bytes,
        samples: Sequence[Union[bytes, bytearray]],
    ) -> DelimiterCandidate:
        """Score a single candidate delimiter against a collection of input samples.

        Evaluates:
        1. Presence Ratio: Fraction of seeds containing candidate.
        2. Frequency Regularity: Typical count per seed (bell curve centered around 2-10).
        3. Field Regularity: Distribution and plausibility of field lengths when split.
        4. Punctuation Prior: Structural framing likelihood based on protocol conventions.

        Args:
            candidate: Delimiter byte sequence to evaluate.
            samples: List of input payloads from the corpus.

        Returns:
            DelimiterCandidate dataclass with detailed metrics and confidence score.
        """
        if not samples:
            return DelimiterCandidate(
                delimiter=candidate,
                confidence=0.0,
                presence_ratio=0.0,
                frequency_per_seed=0.0,
                field_regularity_score=0.0,
                punctuation_prior=0.0,
                avg_field_length=0.0,
                distinct_fields_found=0,
                selection_reason="empty_corpus",
            )

        # Basic validity check
        if any(b in DELIMITER_BLACKLIST for b in candidate):
            return DelimiterCandidate(
                delimiter=candidate,
                confidence=0.0,
                presence_ratio=0.0,
                frequency_per_seed=0.0,
                field_regularity_score=0.0,
                punctuation_prior=0.0,
                avg_field_length=0.0,
                distinct_fields_found=0,
                selection_reason="blacklisted_control_character",
            )

        total_samples = len(samples)
        present_count = 0
        total_occurrences = 0
        field_lengths: List[int] = []
        empty_field_count = 0
        total_fields = 0

        for s in samples:
            b_sample = bytes(s)
            count = b_sample.count(candidate)
            if count > 0:
                present_count += 1
                total_occurrences += count

                # Strip trailing line terminators before splitting for clean field analysis
                stripped = b_sample.rstrip(b"\r\n")
                fields = stripped.split(candidate)
                total_fields += len(fields)
                for f in fields:
                    if len(f) == 0:
                        empty_field_count += 1
                    else:
                        field_lengths.append(len(f))

        # 1. Corpus Presence Ratio: in [0.0, 1.0]
        presence_ratio = present_count / total_samples

        if presence_ratio < self.min_corpus_presence or present_count == 0:
            return DelimiterCandidate(
                delimiter=candidate,
                confidence=0.0,
                presence_ratio=presence_ratio,
                frequency_per_seed=0.0,
                field_regularity_score=0.0,
                punctuation_prior=STANDARD_DELIMITER_PRIORS.get(candidate, 0.3),
                avg_field_length=0.0,
                distinct_fields_found=0,
                selection_reason=f"insufficient_presence ({presence_ratio:.2f} < {self.min_corpus_presence:.2f})",
            )

        # 2. Average Frequency per present seed
        avg_freq = total_occurrences / present_count
        if avg_freq < self.min_occurrences_per_seed:
            freq_score = max(0.0, avg_freq / self.min_occurrences_per_seed)
        else:
            # Ideal field count is 2 to 15 delimiter occurrences.
            # Very high counts (>100) indicate repeated spam / binary runs.
            if avg_freq <= 10.0:
                freq_score = 1.0
            elif avg_freq <= 30.0:
                freq_score = 1.0 - ((avg_freq - 10.0) / 40.0)
            else:
                freq_score = max(0.1, 0.5 - ((avg_freq - 30.0) / 100.0))

        # 3. Field Regularity Score
        # Structured protocols typically have non-empty fields with lengths between 2 and 40 bytes.
        if field_lengths:
            avg_field_len = sum(field_lengths) / len(field_lengths)
            # Penalize if too many empty fields (e.g. repeated "||||")
            empty_ratio = empty_field_count / max(1, total_fields)
            non_empty_penalty = max(0.0, 1.0 - 2.0 * empty_ratio)

            # Plausibility score for field length (ideal: 2 to 32 bytes)
            if 2.0 <= avg_field_len <= 32.0:
                len_score = 1.0
            elif 1.0 <= avg_field_len < 2.0:
                len_score = 0.5
            elif 32.0 < avg_field_len <= 80.0:
                len_score = 1.0 - ((avg_field_len - 32.0) / 60.0)
            else:
                len_score = max(0.1, 0.5 - ((avg_field_len - 80.0) / 200.0))

            field_regularity = len_score * non_empty_penalty
        else:
            avg_field_len = 0.0
            field_regularity = 0.0

        # 4. Punctuation Prior
        prior = STANDARD_DELIMITER_PRIORS.get(candidate, 0.35)

        # Composite Confidence Formula
        # Weights: Presence 35%, Frequency 25%, Field Regularity 25%, Prior 15%
        confidence = (
            0.35 * presence_ratio
            + 0.25 * freq_score
            + 0.25 * field_regularity
            + 0.15 * prior
        )
        confidence = max(0.0, min(1.0, confidence))

        reason = (
            f"presence={presence_ratio:.2f}, freq={avg_freq:.1f}/seed, "
            f"avg_field_len={avg_field_len:.1f}b, prior={prior:.2f}"
        )

        return DelimiterCandidate(
            delimiter=candidate,
            confidence=confidence,
            presence_ratio=presence_ratio,
            frequency_per_seed=avg_freq,
            field_regularity_score=field_regularity,
            punctuation_prior=prior,
            avg_field_length=avg_field_len,
            distinct_fields_found=total_fields,
            selection_reason=reason,
            metrics={
                "presence_ratio": presence_ratio,
                "freq_score": freq_score,
                "field_regularity": field_regularity,
                "prior": prior,
            },
        )

    def detect_delimiters(
        self,
        samples: Sequence[Union[bytes, bytearray]],
    ) -> List[DelimiterCandidate]:
        """Detect and rank all candidate delimiters across the given corpus samples.

        Args:
            samples: List of input payloads.

        Returns:
            List of DelimiterCandidate objects sorted by confidence in descending order.
        """
        candidates: List[DelimiterCandidate] = []
        for cand in self.candidate_bytes:
            scored = self.score_candidate(cand, samples)
            if scored.confidence > 0.0:
                candidates.append(scored)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_best_delimiter(
        self,
        samples: Sequence[Union[bytes, bytearray]],
        fallback: Optional[bytes] = None,
    ) -> Tuple[bytes, float, str, List[DelimiterCandidate]]:
        """Detect the single best delimiter candidate from corpus samples.

        Args:
            samples: List of corpus payloads.
            fallback: Delimiter byte to return if auto-detection fails or confidence < threshold.

        Returns:
            Tuple of (selected_delimiter, confidence, reason_string, all_ranked_candidates).
        """
        if fallback is None:
            fallback = self.default_delimiter

        ranked = self.detect_delimiters(samples)
        if not ranked:
            return fallback, 0.0, "no_valid_candidates_fallback_to_default", []

        best = ranked[0]
        if best.confidence >= self.confidence_threshold:
            return best.delimiter, best.confidence, f"auto_detected: {best.selection_reason}", ranked
        else:
            return (
                fallback,
                best.confidence,
                f"low_confidence_fallback ({best.confidence:.3f} < {self.confidence_threshold:.3f})",
                ranked,
            )

    def rank_candidates(
        self,
        samples: Sequence[Union[bytes, bytearray]],
    ) -> List[DelimiterCandidate]:
        """Rank all candidates across samples."""
        return self.detect_delimiters(samples)

    def detect_from_bytes(
        self,
        samples: Sequence[Union[bytes, bytearray]],
        fallback: Optional[bytes] = None,
    ) -> DelimiterCandidate:
        """Detect the best candidate from a sequence of raw byte samples."""
        if fallback is None:
            fallback = self.default_delimiter
        ranked = self.detect_delimiters(samples)
        if not ranked:
            return DelimiterCandidate(
                delimiter=fallback,
                confidence=0.0,
                presence_ratio=0.0,
                frequency_per_seed=0.0,
                field_regularity_score=0.0,
                punctuation_prior=0.0,
                avg_field_length=0.0,
                distinct_fields_found=0,
                selection_reason="no_candidates_fallback",
            )
        best = ranked[0]
        if best.confidence >= self.confidence_threshold:
            return best
        return DelimiterCandidate(
            delimiter=fallback,
            confidence=0.0,
            presence_ratio=0.0,
            frequency_per_seed=0.0,
            field_regularity_score=0.0,
            punctuation_prior=0.0,
            avg_field_length=0.0,
            distinct_fields_found=0,
            selection_reason=f"low_confidence_fallback ({best.confidence:.3f} < {self.confidence_threshold:.3f})",
        )

    def detect_from_corpus(
        self,
        corpus: Any,
        fallback: Optional[bytes] = None,
    ) -> DelimiterCandidate:
        """Detect the best candidate from a Corpus instance or iterable of seeds."""
        samples: List[bytes] = []
        if hasattr(corpus, "seeds"):
            seeds = getattr(corpus, "seeds")
            if isinstance(seeds, dict):
                seeds = list(seeds.values())
            for s in seeds:
                if hasattr(s, "input_bytes"):
                    samples.append(bytes(s.input_bytes))
                elif isinstance(s, (bytes, bytearray)):
                    samples.append(bytes(s))
        elif hasattr(corpus, "__iter__"):
            for s in corpus:
                if hasattr(s, "input_bytes"):
                    samples.append(bytes(s.input_bytes))
                elif isinstance(s, (bytes, bytearray)):
                    samples.append(bytes(s))
        return self.detect_from_bytes(samples, fallback=fallback)

