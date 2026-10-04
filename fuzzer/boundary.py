"""Structural delimiter boundary detection for Phase 6B.

Identifies valid field and delimiter insertion boundaries in structured
protocols (e.g. pipe-delimited text payloads) without building a full parser.
"""

from typing import Any, List, Optional, Tuple, Union
import random

from fuzzer.delimiter_detector import (
    DelimiterCandidate,
    DelimiterDetector,
    STANDARD_DELIMITER_PRIORS,
)


def find_delimiter_boundaries(
    data: Union[bytes, bytearray],
    delimiter: bytes = b"|",
) -> List[int]:
    """Identify structural delimiter boundaries in the input buffer.

    Calculates valid insertion positions:
    - Index 0 (beginning of input)
    - len(data) (end of input)
    - Positions immediately before trailing newline characters (\\r, \\n)
    - Positions immediately before each delimiter occurrence
    - Positions immediately after each delimiter occurrence

    Args:
        data: Raw byte sequence to inspect.
        delimiter: Byte sequence acting as the field delimiter (default: b"|").

    Returns:
        Sorted list of unique integer indices in [0, len(data)].
    """
    length = len(data)
    if length == 0:
        return [0]

    b = bytes(data)
    positions = {0, length}

    # Handle trailing line terminators if present
    if b.endswith(b"\r\n") and length >= 2:
        positions.add(length - 2)
    elif b.endswith(b"\n") and length >= 1:
        positions.add(length - 1)

    # Find all occurrences of delimiter
    delim_len = len(delimiter)
    if delim_len > 0:
        idx = 0
        while True:
            pos = b.find(delimiter, idx)
            if pos == -1:
                break
            # Position immediately before delimiter
            positions.add(pos)
            # Position immediately after delimiter
            positions.add(pos + delim_len)
            idx = pos + 1

    return sorted(positions)


def format_boundary_insertion(
    buf: Union[bytes, bytearray],
    token: Union[bytes, bytearray],
    pos: int,
    delimiter: bytes = b"|",
) -> Tuple[int, bytes]:
    """Format token insertion at a chosen boundary to prevent malformed syntax.

    Avoids creating duplicate consecutive delimiters (e.g. "||") while ensuring
    inserted fields are properly delimited where required.

    Args:
        buf: Existing buffer bytes.
        token: Dictionary token to insert.
        pos: Target boundary index.
        delimiter: Protocol delimiter (default: b"|").

    Returns:
        Tuple of (clamped_position, formatted_token_bytes).
    """
    b = bytes(buf)
    tok = bytes(token)

    if not b or not tok:
        return max(0, min(pos, len(b))), tok

    clamped_pos = max(0, min(pos, len(b)))
    delim_len = len(delimiter)

    # Context inspection around insertion position
    is_prev_delim = clamped_pos >= delim_len and b[clamped_pos - delim_len : clamped_pos] == delimiter
    is_next_delim = clamped_pos + delim_len <= len(b) and b[clamped_pos : clamped_pos + delim_len] == delimiter
    tok_starts_delim = tok.startswith(delimiter)
    tok_ends_delim = tok.endswith(delimiter)

    # Check whether insertion is at a trailing boundary (end of buffer or right before trailing newline)
    is_trailing_boundary = (
        clamped_pos == len(b)
        or (b.endswith(b"\r\n") and clamped_pos == len(b) - 2)
        or (b.endswith(b"\n") and clamped_pos == len(b) - 1)
    )

    # Rule 1: Token starts with delimiter, but preceding character is already a delimiter
    if tok_starts_delim and is_prev_delim:
        tok = tok[delim_len:]

    # Rule 2: Token does not start with delimiter, but we are appending at a trailing boundary of a non-empty buffer
    elif not tok_starts_delim and not is_prev_delim and is_trailing_boundary:
        tok = delimiter + tok

    # Rule 3: Inserting into middle of buffer immediately after an existing delimiter
    elif not tok_starts_delim and is_prev_delim and not is_next_delim and clamped_pos < len(b):
        if not tok_ends_delim:
            tok = tok + delimiter

    return clamped_pos, tok


class BoundaryDetector:
    """Lightweight structural boundary detector for delimiter-framed protocols."""

    def __init__(self, delimiter: bytes = b"|") -> None:
        """Initialize the boundary detector with a specific protocol delimiter."""
        self.delimiter = delimiter

    def find_boundaries(self, data: Union[bytes, bytearray]) -> List[int]:
        """Return list of valid insertion indices for the input data."""
        return find_delimiter_boundaries(data, delimiter=self.delimiter)

    def has_internal_boundaries(self, data: Union[bytes, bytearray]) -> bool:
        """Check if data contains at least one delimiter occurrence."""
        return self.delimiter in bytes(data)

    def select_boundary(self, data: Union[bytes, bytearray], rng: random.Random) -> int:
        """Select a random boundary position from available boundaries using rng."""
        boundaries = self.find_boundaries(data)
        return rng.choice(boundaries)

    def format_insertion(
        self,
        buf: Union[bytes, bytearray],
        token: Union[bytes, bytearray],
        pos: int,
    ) -> Tuple[int, bytes]:
        """Format token bytes for insertion at pos avoiding duplicate delimiters."""
        return format_boundary_insertion(buf, token, pos, delimiter=self.delimiter)

    def set_delimiter(self, delimiter: Union[bytes, bytearray, str]) -> None:
        """Update the structural delimiter used for boundary calculation."""
        if isinstance(delimiter, str):
            self.delimiter = delimiter.encode("latin1")
        else:
            self.delimiter = bytes(delimiter)

    def auto_detect(
        self,
        samples: List[Union[bytes, bytearray]],
        detector: Optional[Any] = None,
    ) -> Tuple[bytes, float, str]:
        """Auto-detect delimiter from corpus samples and update internal delimiter."""
        from fuzzer.delimiter_detector import DelimiterDetector

        d = detector or DelimiterDetector(default_delimiter=self.delimiter)
        best, conf, reason, _ = d.detect_best_delimiter(samples, fallback=self.delimiter)
        self.set_delimiter(best)
        return best, conf, reason
