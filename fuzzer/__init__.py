"""NeuroFuzz - Coverage-Guided Fuzzer with Contextual Bandit Seed Scheduling."""

from typing import Any

from fuzzer.boundary import (
    BoundaryDetector,
    find_delimiter_boundaries,
    format_boundary_insertion,
)
from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.coverage import CoverageTracker, extract_coverage_depth, parse_coverage_markers
from fuzzer.delimiter_detector import (
    DelimiterCandidate,
    DelimiterDetector,
    STANDARD_DELIMITER_PRIORS,
)
from fuzzer.dictionary import Dictionary
from fuzzer.donor_selector import (
    DonorCompatibility,
    DonorSelectionRecord,
    DonorSelector,
    DonorSelectorTelemetry,
)
from fuzzer.executor import (
    BaseExecutor,
    CrashDetector,
    CrashSaver,
    ExecutionResult,
    Executor,
    SubprocessExecutor,
)
from fuzzer.persistent_executor import PersistentExecutor
from fuzzer.contextual_mutation_bandit import ContextualMutationBandit
from fuzzer.features import FeatureExtractor
from fuzzer.mutation_bandit import MutationBandit, get_default_mutation_arms
from fuzzer.mutation_features import MutationFeatureExtractor
from fuzzer.splicer import FieldSplicer, SplicerTelemetry
from fuzzer.state_frontier import (
    FrontierCandidate,
    FrontierSeed,
    FrontierTelemetry,
    StateFrontier,
)
from fuzzer.mutator import (
    Mutator,
    op_delete_byte,
    op_dictionary_insert,
    op_dictionary_insert_boundary,
    op_dictionary_insert_random,
    op_dictionary_replace,
    op_field_splice,
    op_flip_bit,
    op_insert_byte,
    op_replace_byte,
    op_state_frontier_extend,
)

__all__ = [
    "BaseExecutor",
    "BoundaryDetector",
    "ContextualMutationBandit",
    "Corpus",
    "CoverageTracker",
    "CrashDetector",
    "CrashSaver",
    "DelimiterCandidate",
    "DelimiterDetector",
    "Dictionary",
    "DonorCompatibility",
    "DonorSelectionRecord",
    "DonorSelector",
    "DonorSelectorTelemetry",
    "ExecutionResult",
    "Executor",
    "FrontierCandidate",
    "FrontierSeed",
    "FrontierTelemetry",
    "PersistentExecutor",
    "StateFrontier",
    "SubprocessExecutor",
    "FeatureExtractor",
    "FieldSplicer",
    "Fuzzer",
    "FuzzStats",
    "HeuristicScheduler",
    "LinearBanditScheduler",
    "MutationBandit",
    "MutationFeatureExtractor",
    "Mutator",
    "OnlineLinearModel",
    "RandomScheduler",
    "SeedRecord",
    "SeedScheduler",
    "SplicerTelemetry",
    "STANDARD_DELIMITER_PRIORS",
    "extract_coverage_depth",
    "find_delimiter_boundaries",
    "format_boundary_insertion",
    "get_default_mutation_arms",
    "op_delete_byte",
    "op_dictionary_insert",
    "op_dictionary_insert_boundary",
    "op_dictionary_insert_random",
    "op_dictionary_replace",
    "op_field_splice",
    "op_flip_bit",
    "op_insert_byte",
    "op_replace_byte",
    "op_state_frontier_extend",
    "parse_coverage_markers",
]


def __getattr__(name: str) -> Any:
    """Lazy-load Fuzzer and FuzzStats to avoid runpy circular import warnings."""
    if name == "Fuzzer":
        from fuzzer.fuzzer import Fuzzer
        return Fuzzer
    if name == "FuzzStats":
        from fuzzer.fuzzer import FuzzStats
        return FuzzStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
