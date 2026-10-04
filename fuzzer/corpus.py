"""Corpus management and SeedRecord metadata module for NeuroFuzz.

Maintains raw seed inputs along with scheduling statistics, coverage metadata,
and dynamic seed admission.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Union


@dataclass
class SeedRecord:
    """Encapsulates a corpus seed and its historical performance metrics."""

    id: int
    data: bytes
    name: str
    times_selected: int = 0
    times_mutated: int = 0
    times_produced_coverage: int = 0
    total_coverage_discovered: int = 0
    last_iteration_selected: int = 0
    coverage_units: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Return length of seed data in bytes."""
        return len(self.data)

    def record_selection(self, iteration: int) -> None:
        """Record that this seed was selected for mutation."""
        self.times_selected += 1
        self.times_mutated += 1
        self.last_iteration_selected = iteration

    def record_reward(self, reward: int) -> None:
        """Update yield metrics when a mutation of this seed produces new coverage.

        Args:
            reward: Number of new coverage units discovered (>= 0).
        """
        if reward > 0:
            self.times_produced_coverage += 1
            self.total_coverage_discovered += reward


class Corpus:
    """Manages an active collection of seed records for the fuzzer."""

    def __init__(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize the corpus, optionally loading from a directory.

        Args:
            corpus_dir: Optional path to a directory containing seed files.
            rng: Optional random.Random instance for deterministic behavior.
        """
        self.rng = rng or random.Random()
        self.corpus_dir: Optional[Path] = Path(corpus_dir).resolve() if corpus_dir else None
        self.records: List[SeedRecord] = []

        if self.corpus_dir is not None:
            self.load_directory(self.corpus_dir)

    @property
    def seeds(self) -> List[bytes]:
        """Backwards-compatible view of raw seed bytes."""
        return [rec.data for rec in self.records]

    @property
    def seed_names(self) -> List[str]:
        """Backwards-compatible view of seed names."""
        return [rec.name for rec in self.records]

    @property
    def seed_metadata(self) -> List[Dict[str, Any]]:
        """Backwards-compatible view of seed metadata dictionaries."""
        return [rec.metadata for rec in self.records]

    def load_directory(self, corpus_dir: Union[str, Path]) -> None:
        """Load all non-hidden seed files from the given directory as bytes.

        Args:
            corpus_dir: Directory path containing seed files.

        Raises:
            FileNotFoundError: If corpus_dir does not exist.
            ValueError: If corpus_dir is not a directory or contains no valid files.
        """
        path = Path(corpus_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {path}")
        if not path.is_dir():
            raise ValueError(f"Corpus path is not a directory: {path}")

        loaded = 0
        for entry in sorted(path.iterdir()):
            if entry.is_file() and not entry.name.startswith(".") and not entry.name.endswith("_meta.json"):
                content = entry.read_bytes()
                self.add_seed(
                    content,
                    name=entry.name,
                    metadata={"source": "initial_seed", "file": entry.name},
                )
                loaded += 1

        if loaded == 0:
            raise ValueError(f"No seed files found in directory: {path}")

    def add_seed(
        self,
        data: bytes,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        coverage_units: Optional[Set[str]] = None,
    ) -> SeedRecord:
        """Add a seed input to the corpus and create a SeedRecord.

        Args:
            data: Raw byte content of the seed.
            name: Optional descriptive name or filename for the seed.
            metadata: Optional dictionary storing discovery metadata.
            coverage_units: Optional set of coverage units known for this seed.

        Returns:
            The newly created SeedRecord.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Seed data must be bytes or bytearray, got {type(data).__name__}")

        seed_bytes = bytes(data)
        seed_id = len(self.records)
        seed_name = name or f"seed_{seed_id}"
        meta = metadata or {}

        record = SeedRecord(
            id=seed_id,
            data=seed_bytes,
            name=seed_name,
            coverage_units=set(coverage_units) if coverage_units else set(),
            metadata=meta,
        )
        self.records.append(record)
        return record

    def add_interesting_input(
        self,
        data: bytes,
        coverage_units: Iterable[Union[int, str]],
        iteration: int,
        save_to_disk: bool = True,
    ) -> Optional[Path]:
        """Add an input that discovered new coverage to the corpus.

        Args:
            data: Raw byte content of the interesting input.
            coverage_units: The newly discovered coverage units attributed to this input.
            iteration: The fuzzing iteration where it was discovered.
            save_to_disk: Whether to persist the new seed to `self.corpus_dir`.

        Returns:
            Optional Path to the saved file on disk (if save_to_disk=True and corpus_dir exists).
        """
        data_hash = hashlib.sha256(data).hexdigest()
        short_hash = data_hash[:8]
        cov_list = sorted(str(u) for u in coverage_units)
        name = f"input_iter{iteration:06d}_cov{len(cov_list)}_{short_hash}"

        meta = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_coverage_count": len(cov_list),
            "new_coverage_units": cov_list,
            "sha256": data_hash,
            "size_bytes": len(data),
            "source": "coverage_guided_mutation",
        }

        self.add_seed(
            data=data,
            name=name,
            metadata=meta,
            coverage_units=set(cov_list),
        )

        saved_path: Optional[Path] = None
        if save_to_disk and self.corpus_dir is not None:
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            saved_path = self.corpus_dir / f"{name}.bin"
            saved_path.write_bytes(data)

            meta_path = self.corpus_dir / f"{name}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return saved_path

    def select_seed(self) -> bytes:
        """Select a seed from the corpus uniformly at random (legacy API).

        Returns:
            A byte sequence representing the chosen seed.

        Raises:
            ValueError: If the corpus is empty.
        """
        if not self.records:
            raise ValueError("Corpus is empty. Please add or load at least one seed.")
        return self.rng.choice(self.records).data

    def get_record(self, index: int) -> SeedRecord:
        """Access a SeedRecord by index."""
        return self.records[index]

    def __len__(self) -> int:
        """Return the number of seeds currently in the corpus."""
        return len(self.records)

    def __getitem__(self, index: int) -> bytes:
        """Access raw seed bytes by index (backwards-compatible)."""
        return self.records[index].data

    def __repr__(self) -> str:
        return f"<Corpus size={len(self.records)} dir={self.corpus_dir}>"
