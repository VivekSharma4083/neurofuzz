"""Dictionary management for protocol-aware fuzzing in NeuroFuzz (Phase 6A).

Provides clean abstraction for loading, parsing, and selecting tokens
from protocol dictionary files or in-memory sequences.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union
import codecs
import random


class Dictionary:
    """Protocol dictionary containing domain-specific byte tokens for mutation guidance."""

    def __init__(
        self,
        tokens: Optional[Sequence[Union[bytes, str]]] = None,
        filepath: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize a Dictionary.

        Args:
            tokens: Optional sequence of initial token strings or bytes.
            filepath: Optional path to a dictionary file to load immediately.
        """
        self._tokens: List[bytes] = []
        self._token_set: Set[bytes] = set()

        if filepath is not None:
            self.load_from_file(filepath)

        if tokens is not None:
            for tok in tokens:
                self.add_token(tok)

    def add_token(self, token: Union[bytes, str]) -> None:
        """Add a single token to the dictionary, avoiding duplicates.

        Args:
            token: String or byte sequence representing the token.
        """
        if isinstance(token, str):
            token_bytes = token.encode("utf-8")
        elif isinstance(token, (bytes, bytearray)):
            token_bytes = bytes(token)
        else:
            raise TypeError(f"Token must be bytes or str, got {type(token).__name__}")

        if not token_bytes:
            return

        if token_bytes not in self._token_set:
            self._token_set.add(token_bytes)
            self._tokens.append(token_bytes)

    def load_from_file(self, filepath: Union[str, Path]) -> "Dictionary":
        """Load and parse tokens from an external dictionary file.

        Lines starting with '#' and blank lines are ignored.
        Tokens may optionally be enclosed in double or single quotes with escape sequences.
        Whitespace is stripped.

        Args:
            filepath: Path to the dictionary file.

        Returns:
            self for chaining.

        Raises:
            FileNotFoundError: If filepath does not exist.
        """
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dictionary file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # Handle quoted strings e.g. "USER=admin" or '\x00\x01'
                if (stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2) or (
                    stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2
                ):
                    raw_content = stripped[1:-1]
                    try:
                        token_bytes = codecs.escape_decode(raw_content.encode("utf-8"))[0]
                    except Exception:
                        token_bytes = raw_content.encode("utf-8")
                else:
                    token_bytes = stripped.encode("utf-8")

                self.add_token(token_bytes)

        return self

    @property
    def tokens(self) -> List[bytes]:
        """Return a copy of all loaded byte tokens."""
        return list(self._tokens)

    def get_random_token(self, rng: random.Random) -> bytes:
        """Select a random token using the provided RNG.

        Args:
            rng: random.Random instance for deterministic reproduction.

        Returns:
            A byte token.

        Raises:
            ValueError: If the dictionary is empty.
        """
        if not self._tokens:
            raise ValueError("Cannot select token from an empty dictionary.")
        return rng.choice(self._tokens)

    def __len__(self) -> int:
        """Return total number of unique tokens in the dictionary."""
        return len(self._tokens)

    def __iter__(self):
        """Iterate over byte tokens."""
        return iter(self._tokens)

    def __contains__(self, item: Union[bytes, str]) -> bool:
        """Check if a token exists in the dictionary."""
        if isinstance(item, str):
            item = item.encode("utf-8")
        return item in self._token_set

    def __repr__(self) -> str:
        return f"Dictionary(tokens={len(self._tokens)})"
