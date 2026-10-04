"""Mutation engine for NeuroFuzz (Phase 6B).

Implements modular byte-level, protocol-aware dictionary, and delimiter
boundary-aware mutation operators:
- Bit flip
- Byte replacement
- Byte insertion
- Byte deletion
- Dictionary token insertion (random mode - Phase 6A)
- Dictionary token insertion (boundary-aware mode - Phase 6B)
- Dictionary token replacement (Phase 6A)
"""

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fuzzer.boundary import BoundaryDetector
from fuzzer.dictionary import Dictionary
from fuzzer.donor_selector import DonorSelector
from fuzzer.splicer import FieldSplicer

MutationOperator = Callable[[bytearray, random.Random], bytearray]
DictionaryMutationOperator = Callable[..., bytearray]


def op_flip_bit(buf: bytearray, rng: random.Random) -> bytearray:
    """Flip a single random bit in the bytearray."""
    if not buf:
        # If input is empty, seed it with a random byte
        buf.append(rng.randint(0, 255))
        return buf

    byte_idx = rng.randrange(len(buf))
    bit_idx = rng.randrange(8)
    buf[byte_idx] ^= (1 << bit_idx)
    return buf


def op_replace_byte(buf: bytearray, rng: random.Random) -> bytearray:
    """Replace a random byte with a random value (0..255)."""
    if not buf:
        buf.append(rng.randint(0, 255))
        return buf

    byte_idx = rng.randrange(len(buf))
    current_val = buf[byte_idx]
    # Pick a new value, trying to ensure it differs from current
    new_val = rng.randint(0, 255)
    if new_val == current_val:
        new_val = (current_val + rng.randint(1, 255)) % 256
    buf[byte_idx] = new_val
    return buf


def op_insert_byte(buf: bytearray, rng: random.Random) -> bytearray:
    """Insert a random byte at a random index."""
    idx = rng.randint(0, len(buf))
    val = rng.randint(0, 255)
    buf.insert(idx, val)
    return buf


def op_delete_byte(buf: bytearray, rng: random.Random) -> bytearray:
    """Delete a random byte from the bytearray."""
    if not buf:
        return buf
    idx = rng.randrange(len(buf))
    del buf[idx]
    return buf


def op_dictionary_insert(
    buf: bytearray,
    rng: random.Random,
    dictionary: Optional[Dictionary] = None,
    boundary_aware: bool = False,
    detector: Optional[BoundaryDetector] = None,
) -> bytearray:
    """Insert a token from the dictionary into the buffer.

    Args:
        buf: Mutable bytearray to mutate in-place.
        rng: random.Random instance for deterministic randomness.
        dictionary: Optional protocol Dictionary.
        boundary_aware: If True, inserts only at detected field boundaries (Phase 6B).
            If False, inserts at an arbitrary random byte offset (Phase 6A).
        detector: Optional BoundaryDetector instance.

    Returns:
        The mutated bytearray buffer.
    """
    if dictionary is None or len(dictionary) == 0:
        return op_insert_byte(buf, rng)

    token = dictionary.get_random_token(rng)
    if not buf:
        buf.extend(token)
        return buf

    if boundary_aware:
        det = detector or BoundaryDetector()
        pos = det.select_boundary(buf, rng)
        pos, formatted_token = det.format_insertion(buf, token, pos)
        buf[pos:pos] = formatted_token
        return buf
    else:
        idx = rng.randint(0, len(buf))
        buf[idx:idx] = token
        return buf


def op_dictionary_insert_random(
    buf: bytearray,
    rng: random.Random,
    dictionary: Optional[Dictionary] = None,
) -> bytearray:
    """Insert a dictionary token at an arbitrary byte offset (Phase 6A)."""
    return op_dictionary_insert(buf, rng, dictionary=dictionary, boundary_aware=False)


def op_dictionary_insert_boundary(
    buf: bytearray,
    rng: random.Random,
    dictionary: Optional[Dictionary] = None,
    detector: Optional[BoundaryDetector] = None,
) -> bytearray:
    """Insert a dictionary token at a detected delimiter boundary (Phase 6B)."""
    return op_dictionary_insert(buf, rng, dictionary=dictionary, boundary_aware=True, detector=detector)


def op_dictionary_replace(
    buf: bytearray,
    rng: random.Random,
    dictionary: Optional[Dictionary] = None,
) -> bytearray:
    """Replace an existing token or byte sequence with a dictionary token.

    If any multi-character dictionary tokens are present in buf, selects one
    and replaces it with another token from the dictionary.
    Otherwise, overwrites bytes at a random position with a dictionary token.
    If the dictionary is None or empty, falls back to op_replace_byte.
    """
    if dictionary is None or len(dictionary) == 0:
        return op_replace_byte(buf, rng)

    token = dictionary.get_random_token(rng)
    if not buf:
        buf.extend(token)
        return buf

    # Check for known dictionary tokens in the buffer
    known_tokens = [tok for tok in dictionary.tokens if len(tok) >= 2 and tok in buf]
    if known_tokens:
        target_tok = rng.choice(known_tokens)
        indices: List[int] = []
        pos = 0
        while True:
            idx = buf.find(target_tok, pos)
            if idx == -1:
                break
            indices.append(idx)
            pos = idx + 1

        chosen_idx = rng.choice(indices)
        buf[chosen_idx : chosen_idx + len(target_tok)] = token
        return buf

    # Fallback: overwrite at random byte offset
    idx = rng.randint(0, max(0, len(buf) - 1))
    replace_len = min(len(token), len(buf) - idx)
    buf[idx : idx + replace_len] = token[:replace_len]
    return buf


def op_field_splice(
    buf: bytearray,
    rng: random.Random,
    splicer: Optional[FieldSplicer] = None,
    donor: Optional[bytes] = None,
) -> bytearray:
    """Recombine fields from buffer with fields from donor using FieldSplicer.

    Args:
        buf: Mutable bytearray of Parent A to mutate in-place.
        rng: random.Random instance.
        splicer: Optional FieldSplicer instance.
        donor: Byte payload of donor Parent B.

    Returns:
        The spliced bytearray buffer (or fallback mutation if splicing was rejected).
    """
    if not isinstance(buf, bytearray):
        buf = bytearray(buf)
    if splicer is None or donor is None:
        return op_flip_bit(buf, rng)

    result = splicer.splice(bytes(buf), donor)
    if result is not None and result != bytes(buf):
        buf.clear()
        buf.extend(result)
        return buf

def op_state_frontier_extend(
    buf: bytearray,
    rng: random.Random,
    dictionary: Optional[Dictionary] = None,
    detector: Optional[BoundaryDetector] = None,
) -> bytearray:
    """State-frontier boundary extension fallback operator (Phase 7C)."""
    return op_dictionary_insert_boundary(buf, rng, dictionary=dictionary, detector=detector)


class Mutator:
    """Modular mutation engine supporting byte-level, dictionary, and boundary-aware operators."""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        dictionary: Optional[Dictionary] = None,
        dictionary_probability: float = 0.0,
        boundary_aware: bool = False,
        delimiter: bytes = b"|",
        splicer: Optional[FieldSplicer] = None,
        enable_splicing: bool = False,
        splice_probability: float = 0.0,
        max_splice_size: int = 512,
        corpus_provider: Optional[Callable[[], Sequence[Any]]] = None,
        donor_selector: Optional[DonorSelector] = None,
        donor_policy: str = "random",
    ) -> None:
        """Initialize the Mutator.

        Args:
            rng: Optional random.Random instance for deterministic mutations.
            dictionary: Optional protocol Dictionary containing domain tokens.
            dictionary_probability: Probability in [0.0, 1.0] of choosing a dictionary
                mutation over a byte-level mutation. Default 0.0 (disabled).
            boundary_aware: If True, dictionary insertions snap to structural delimiter
                boundaries (Phase 6B). Default False (Phase 6A random insertion).
            delimiter: Structural protocol delimiter byte (default: b"|").
            splicer: Optional FieldSplicer instance for structured crossover (Phase 6E).
            enable_splicing: If True, enables field_splice operator.
            splice_probability: Probability in [0.0, 1.0] of selecting field_splice in fixed mode.
            max_splice_size: Maximum byte length for spliced candidates (default: 512).
            corpus_provider: Optional zero-arg callable returning sequence of corpus seed bytes or records.
            donor_selector: Optional DonorSelector instance for compatibility-aware donor selection (Phase 6F).
            donor_policy: Donor selection strategy ('random' or 'compatible'). Default: 'random'.
        """
        self.rng = rng or random.Random()
        self.dictionary = dictionary
        if not (0.0 <= dictionary_probability <= 1.0):
            raise ValueError(
                f"dictionary_probability must be in [0.0, 1.0], got {dictionary_probability}"
            )
        self.dictionary_probability = dictionary_probability
        self.boundary_aware = boundary_aware
        self.boundary_detector = BoundaryDetector(delimiter=delimiter)

        # Structured crossover & field splicing (Phase 6E / 6F)
        self.enable_splicing = enable_splicing or (splicer is not None)
        self.splice_probability = float(splice_probability)
        self.max_splice_size = int(max_splice_size)
        self.corpus_provider = corpus_provider
        self.splicer = (
            splicer
            if splicer is not None
            else (
                FieldSplicer(
                    delimiter=delimiter,
                    max_splice_size=self.max_splice_size,
                    rng=self.rng,
                )
                if self.enable_splicing
                else None
            )
        )
        self.donor_selector = (
            donor_selector
            if donor_selector is not None
            else (
                DonorSelector(policy=donor_policy, delimiter=delimiter, rng=self.rng)
                if self.enable_splicing
                else None
            )
        )

        self._operators: Dict[str, Callable] = {}
        self._byte_operator_names: List[str] = []
        self._dict_operator_names: List[str] = []
        self._all_operator_names: List[str] = []
        self.last_operator: str = ""
        self.last_donor: Optional[bytes] = None

        # Register default byte-level operators
        self.register_operator("flip_bit", op_flip_bit, is_dictionary_operator=False)
        self.register_operator("replace_byte", op_replace_byte, is_dictionary_operator=False)
        self.register_operator("insert_byte", op_insert_byte, is_dictionary_operator=False)
        self.register_operator("delete_byte", op_delete_byte, is_dictionary_operator=False)

        # Register protocol dictionary operators
        self.register_operator("dictionary_insert", op_dictionary_insert, is_dictionary_operator=True)
        self.register_operator("dictionary_insert_random", op_dictionary_insert_random, is_dictionary_operator=True)
        self.register_operator("dictionary_insert_boundary", op_dictionary_insert_boundary, is_dictionary_operator=True)
        self.register_operator("dictionary_replace", op_dictionary_replace, is_dictionary_operator=True)
        self.register_operator("state_frontier_extend", op_state_frontier_extend, is_dictionary_operator=True)

        # Register field splice operator if enabled (Phase 6E)
        if self.enable_splicing:
            self.register_operator("field_splice", op_field_splice, is_dictionary_operator=False)

    def register_operator(
        self,
        name: str,
        operator_fn: Callable,
        is_dictionary_operator: bool = False,
    ) -> None:
        """Register a new mutation operator."""
        if name not in self._all_operator_names:
            self._all_operator_names.append(name)
        if is_dictionary_operator:
            if name not in self._dict_operator_names:
                self._dict_operator_names.append(name)
        else:
            if name not in self._byte_operator_names:
                self._byte_operator_names.append(name)
        self._operators[name] = operator_fn

    def set_delimiter(self, delimiter: Union[bytes, bytearray, str]) -> None:
        """Update the protocol delimiter across boundary detector and splicing components."""
        if isinstance(delimiter, str):
            delim = delimiter.encode("latin1")
        else:
            delim = bytes(delimiter)
        self.boundary_detector.set_delimiter(delim)
        if self.splicer is not None and hasattr(self.splicer, "delimiter"):
            self.splicer.delimiter = delim
        if self.donor_selector is not None and hasattr(self.donor_selector, "delimiter"):
            self.donor_selector.delimiter = delim

    @property
    def operators(self) -> List[str]:
        """Return list of all registered operator names."""
        return list(self._all_operator_names)

    @property
    def byte_operators(self) -> List[str]:
        """Return list of registered byte-level operator names."""
        return list(self._byte_operator_names)

    @property
    def dictionary_operators(self) -> List[str]:
        """Return list of registered dictionary operator names."""
        return list(self._dict_operator_names)

    def mutate_with_operator(
        self,
        data: bytes,
        operator_name: Optional[str] = None,
        donor: Optional[bytes] = None,
    ) -> Tuple[bytes, str]:
        """Apply a mutation to a copy of the input data and return both result and operator name.

        Args:
            data: Original byte sequence (never modified in-place).
            operator_name: Optional specific operator to invoke. If None, chosen
                according to dictionary_probability, splice_probability, and settings.
            donor: Optional donor bytes for field_splice. If None, resolved via corpus_provider.

        Returns:
            Tuple of (mutated_bytes, operator_name_used).
        """
        if not self._all_operator_names:
            raise RuntimeError("No mutation operators registered.")

        if operator_name is None:
            if (
                self.enable_splicing
                and self.splice_probability > 0.0
                and self.rng.random() < self.splice_probability
            ):
                operator_name = "field_splice"
            elif (
                self.dictionary is not None
                and len(self.dictionary) > 0
                and self.dictionary_probability > 0.0
                and self.rng.random() < self.dictionary_probability
            ):
                # When dictionary mutation is selected, choose between insertion and replacement
                # In boundary-aware mode, use dictionary_insert_boundary; otherwise dictionary_insert_random
                insert_op = "dictionary_insert_boundary" if self.boundary_aware else "dictionary_insert_random"
                operator_name = self.rng.choice([insert_op, "dictionary_replace"])
            else:
                # Byte-level operators (excluding field_splice unless explicitly chosen)
                base_byte_ops = [op for op in self._byte_operator_names if op != "field_splice"]
                operator_name = self.rng.choice(base_byte_ops)
        elif operator_name not in self._operators:
            raise KeyError(f"Unknown mutation operator: {operator_name}")

        buf = bytearray(data)
        self.last_donor = None

        # Dispatch operator with appropriate arguments
        if operator_name == "field_splice":
            chosen_donor = donor
            if chosen_donor is None and self.corpus_provider is not None:
                corpus_items = self.corpus_provider()
                if self.donor_selector is not None:
                    chosen_donor = self.donor_selector.select_donor(data, corpus_items)
                elif self.splicer is not None:
                    chosen_donor = self.splicer.select_donor(corpus_items, data)

            self.last_donor = chosen_donor
            if chosen_donor is not None and self.splicer is not None:
                spliced_bytes = self.splicer.splice(data, chosen_donor)
                if spliced_bytes is not None and spliced_bytes != data:
                    self.last_operator = "field_splice"
                    return spliced_bytes, "field_splice"

            # Graceful fallback if splicing could not produce a valid candidate
            if self.boundary_aware and self.dictionary is not None and len(self.dictionary) > 0:
                mutated_buf = op_dictionary_insert_boundary(
                    buf, self.rng, self.dictionary, detector=self.boundary_detector
                )
                self.last_operator = "dictionary_insert_boundary"
                return bytes(mutated_buf), "dictionary_insert_boundary"
            else:
                mutated_buf = op_replace_byte(buf, self.rng)
                self.last_operator = "replace_byte"
                return bytes(mutated_buf), "replace_byte"

        elif operator_name == "dictionary_insert_boundary":
            mutated_buf = op_dictionary_insert_boundary(
                buf, self.rng, self.dictionary, detector=self.boundary_detector
            )
        elif operator_name == "dictionary_insert_random":
            mutated_buf = op_dictionary_insert_random(buf, self.rng, self.dictionary)
        elif operator_name == "dictionary_insert":
            resolved_op = "dictionary_insert_boundary" if self.boundary_aware else "dictionary_insert_random"
            operator_name = resolved_op
            mutated_buf = op_dictionary_insert(
                buf,
                self.rng,
                self.dictionary,
                boundary_aware=self.boundary_aware,
                detector=self.boundary_detector,
            )
        elif operator_name == "dictionary_replace":
            mutated_buf = op_dictionary_replace(buf, self.rng, self.dictionary)
        elif operator_name == "state_frontier_extend":
            mutated_buf = op_state_frontier_extend(
                buf, self.rng, self.dictionary, detector=self.boundary_detector
            )
        elif operator_name in self._dict_operator_names:
            mutated_buf = self._operators[operator_name](buf, self.rng, self.dictionary)
        else:
            mutated_buf = self._operators[operator_name](buf, self.rng)

        self.last_operator = operator_name
        return bytes(mutated_buf), operator_name

    def mutate(
        self,
        data: bytes,
        operator_name: Optional[str] = None,
    ) -> bytes:
        """Apply a mutation to a copy of the input data.

        Guarantees that the input `data` is never modified in-place.

        Args:
            data: Original byte sequence.
            operator_name: Optional specific operator to invoke. If None, chosen randomly.

        Returns:
            A new mutated bytes object.
        """
        mutated, _ = self.mutate_with_operator(data, operator_name=operator_name)
        return mutated

    def mutate_chain(self, data: bytes, count: int = 1) -> bytes:
        """Apply multiple sequential mutations to the input data.

        Args:
            data: Original byte sequence.
            count: Number of mutation steps to apply.

        Returns:
            A new mutated bytes object.
        """
        res = data
        for _ in range(max(1, count)):
            res = self.mutate(res)
        return res
