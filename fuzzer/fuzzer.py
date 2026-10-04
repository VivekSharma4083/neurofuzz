"""Main fuzzing engine and CLI runner for NeuroFuzz Phase 3.

Orchestrates seed selection via pluggable seed schedulers (Random or Heuristic),
mutation, target execution, coverage tracking, reward calculation, and telemetry.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Union

from fuzzer.contextual_mutation_bandit import ContextualMutationBandit
from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.coverage import CoverageTracker, extract_coverage_depth
from fuzzer.dictionary import Dictionary
from fuzzer.donor_selector import DonorSelector
from fuzzer.executor import CrashSaver, ExecutionResult, Executor, SubprocessExecutor
from fuzzer.persistent_executor import PersistentExecutor
from fuzzer.mutation_bandit import MutationBandit, get_default_mutation_arms
from fuzzer.mutation_features import MutationFeatureExtractor
from fuzzer.mutator import Mutator
from fuzzer.scheduler import (
    HeuristicScheduler,
    LinearBanditScheduler,
    RandomScheduler,
    SeedScheduler,
)
from fuzzer.splicer import FieldSplicer
from fuzzer.state_frontier import FrontierTelemetry, StateFrontier


@dataclass
class FuzzStats:
    """Telemetry data captured during fuzzing."""

    total_executions: int = 0
    total_mutations: int = 0
    total_coverage_units: int = 0
    new_coverage_discoveries: int = 0
    corpus_size: int = 0
    crashes_found: int = 0
    unique_crashes: int = 0
    timeouts_found: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    scheduler_name: str = ""
    executor_type: str = "subprocess"
    total_reward: int = 0
    average_reward: float = 0.0
    exploration_rate: float = 0.0
    deepest_state_reached: int = 0
    scheduler_telemetry: Dict[str, Any] = field(default_factory=dict)

    # Phase 6A: Dictionary mutation telemetry
    byte_mutations: int = 0
    dictionary_mutations: int = 0
    dictionary_insertions: int = 0
    dictionary_replacements: int = 0
    dictionary_new_coverage: int = 0
    dictionary_crashes: int = 0

    # Phase 6B: Boundary-aware dictionary mutation telemetry
    boundary_aware: bool = False
    dictionary_insert_random: int = 0
    dictionary_insert_boundary: int = 0
    boundary_mutations: int = 0
    boundary_new_coverage: int = 0
    boundary_crashes: int = 0
    total_detected_boundaries: int = 0
    usable_boundary_seeds_count: int = 0
    depth_transition_counts: Dict[str, int] = field(
        default_factory=lambda: {"14->15": 0, "15->16": 0, "16->17": 0, "17->18": 0, "18->19": 0}
    )

    # Phase 6C: UCB1 learned mutation operator selection
    mutation_policy: str = "fixed"
    ucb_c: float = 1.0
    mutation_bandit_telemetry: Dict[str, Any] = field(default_factory=dict)

    # Phase 6D: Contextual mutation operator selection
    mutation_epsilon: float = 0.20
    contextual_mutation_bandit_telemetry: Dict[str, Any] = field(default_factory=dict)

    # Phase 6E: Structured crossover & field splicing telemetry
    enable_splicing: bool = False
    splice_mutations: int = 0
    splice_new_coverage: int = 0
    splice_crashes: int = 0
    splice_telemetry: Dict[str, Any] = field(default_factory=dict)

    # Phase 6F: Compatibility-aware parent selection telemetry
    donor_policy: str = "random"
    donor_telemetry: Dict[str, Any] = field(default_factory=dict)
    compatible_new_coverage: int = 0

    # Phase 7C: State-frontier extension telemetry
    enable_frontier_extension: bool = False
    frontier_extension_attempts: int = 0
    frontier_extension_discoveries: int = 0
    frontier_extension_new_coverage: int = 0
    frontier_extension_depth_discoveries: int = 0
    frontier_telemetry: Dict[str, Any] = field(default_factory=dict)

    # Phase 7D: Adaptive frontier rate & automated delimiter discovery telemetry
    frontier_rate_mode: str = "fixed"
    current_frontier_rate: float = 0.20
    delimiter_mode: str = "static"
    active_delimiter: str = "|"
    delimiter_confidence: float = 1.0
    delimiter_selection_reason: str = "configured_static"
    delimiter_candidates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def dictionary_mutation_rate(self) -> float:
        """Return proportion of mutations that used dictionary operators."""
        return self.dictionary_mutations / max(1, self.total_mutations)

    @property
    def boundary_mutation_rate(self) -> float:
        """Return proportion of dictionary mutations that were boundary-aware."""
        return self.dictionary_insert_boundary / max(1, self.dictionary_mutations)

    @property
    def elapsed_time(self) -> float:
        """Return total elapsed time in seconds."""
        if self.end_time > 0.0:
            return max(0.0001, self.end_time - self.start_time)
        return max(0.0001, time.time() - self.start_time)

    @property
    def exec_per_sec(self) -> float:
        """Return average execution rate (execs/second)."""
        return self.total_executions / self.elapsed_time


class Fuzzer:
    """Phase 3 coverage-guided fuzzer with intelligent seed scheduling."""

    def __init__(
        self,
        target_path: Union[str, Path],
        corpus_dir: Union[str, Path] = "corpus",
        crashes_dir: Union[str, Path] = "crashes",
        iterations: int = 1000,
        timeout: float = 1.0,
        seed: Optional[int] = None,
        stats_interval: int = 50,
        calibrate: bool = True,
        scheduler: Optional[SeedScheduler] = None,
        scheduler_type: str = "random",
        epsilon: float = 0.2,
        learning_rate: float = 0.05,
        l2_reg: float = 0.001,
        dictionary_path: Optional[Union[str, Path]] = None,
        dictionary_probability: float = 0.0,
        dictionary: Optional[Dictionary] = None,
        boundary_aware: bool = False,
        mutation_policy: str = "fixed",
        ucb_c: float = 1.0,
        mutation_bandit: Optional[MutationBandit] = None,
        contextual_mutation_bandit: Optional[ContextualMutationBandit] = None,
        mutation_epsilon: float = 0.20,
        mutation_learning_rate: float = 0.05,
        mutation_l2_reg: float = 0.001,
        enable_splicing: bool = False,
        splice_probability: float = 0.0,
        max_splice_size: int = 512,
        splicer: Optional[FieldSplicer] = None,
        donor_policy: str = "random",
        donor_prefix_weight: float = 3.0,
        donor_command_weight: float = 2.0,
        donor_protocol_weight: float = 1.0,
        donor_extra_field_weight: float = 1.0,
        donor_field_diff_penalty: float = 0.5,
        donor_depth_diff_penalty: float = 0.25,
        donor_selector: Optional[DonorSelector] = None,
        executor_type: str = "subprocess",
        iteration_callback: Optional[Callable[[int, FuzzStats, Dict[str, Any]], None]] = None,
        enable_frontier_extension: bool = False,
        frontier_extension_rate: float = 0.20,
        frontier_rate_mode: str = "fixed",
        min_frontier_rate: float = 0.02,
        max_frontier_rate: float = 0.30,
        frontier_decay_factor: float = 0.85,
        frontier_stagnation_threshold: int = 25,
        delimiter: Union[str, bytes] = "|",
        delimiter_mode: str = "static",
        state_frontier: Optional[StateFrontier] = None,
    ) -> None:
        """Initialize the fuzzer with decoupled components.

        Args:
            target_path: Executable binary to fuzz.
            corpus_dir: Directory containing initial seed files.
            crashes_dir: Directory where crashing inputs are saved.
            iterations: Maximum number of test executions.
            timeout: Subprocess timeout per execution (in seconds).
            seed: Optional RNG seed for reproducibility.
            stats_interval: How often (in iterations) to display progress.
            calibrate: Whether to run initial seeds first to establish baseline coverage.
            scheduler: Optional custom SeedScheduler instance.
            scheduler_type: Scheduler to construct ("random", "heuristic", or "linear_bandit").
            epsilon: Exploration probability for bandit/heuristic scheduler (default: 0.2).
            learning_rate: Step size eta for online linear model updates (default: 0.05).
            l2_reg: L2 regularization coefficient for online linear model (default: 0.001).
            dictionary_path: Optional path to protocol tokens dictionary file (Phase 6A).
            dictionary_probability: Probability of using dictionary mutations in [0.0, 1.0].
            dictionary: Optional pre-constructed Dictionary instance.
            boundary_aware: If True, dictionary insertions snap to structural delimiter boundaries (Phase 6B).
            mutation_policy: Operator selection policy ("fixed", "ucb1", or "contextual") (Phase 6C/6D).
            ucb_c: Exploration parameter for UCB1 mutation bandit (default: 1.0).
            mutation_bandit: Optional pre-constructed MutationBandit instance.
            contextual_mutation_bandit: Optional pre-constructed ContextualMutationBandit instance.
            mutation_epsilon: Exploration rate for contextual mutation bandit (default: 0.20).
            mutation_learning_rate: Learning rate for contextual mutation bandit models (default: 0.05).
            mutation_l2_reg: L2 regularization coefficient for contextual mutation bandit models (default: 0.001).
            enable_splicing: If True, enables structured field splicing / crossover (Phase 6E).
            splice_probability: Probability of choosing field_splice in fixed mode (default: 0.0).
            max_splice_size: Maximum allowed byte length for spliced candidates (default: 512).
            splicer: Optional pre-constructed FieldSplicer instance.
            donor_policy: Donor selection policy ("random" or "compatible") (Phase 6F).
            donor_prefix_weight: Compatibility weight for matching field prefix length (default: 3.0).
            donor_command_weight: Compatibility bonus for identical command field (default: 2.0).
            donor_protocol_weight: Compatibility bonus for identical protocol header (default: 1.0).
            donor_extra_field_weight: Compatibility weight for donor having additional fields (default: 1.0).
            donor_field_diff_penalty: Compatibility penalty per field count difference (default: 0.5).
            donor_depth_diff_penalty: Compatibility penalty per coverage depth difference (default: 0.25).
            donor_selector: Optional pre-constructed DonorSelector instance.
            executor_type: Execution harness ("subprocess" or "persistent").
            iteration_callback: Optional callable invoked after each execution iteration with (iteration, stats, info_dict).
        """
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.iterations = iterations
        self.timeout = timeout
        self.stats_interval = stats_interval
        self.calibrate = calibrate
        self.boundary_aware = boundary_aware
        self.iteration_callback = iteration_callback

        # Corpus initialization before components
        self.corpus = Corpus(corpus_dir=corpus_dir, rng=self.rng)

        # Protocol delimiter configuration and auto-discovery (Phase 7D)
        self.delimiter = delimiter.encode("latin1") if isinstance(delimiter, str) else bytes(delimiter)
        self.delimiter_mode = delimiter_mode.lower()
        self.delimiter_confidence: float = 1.0
        self.delimiter_selection_reason: str = "configured_static"
        self.delimiter_candidates: List[Dict[str, Any]] = []

        if self.delimiter_mode == "auto":
            from fuzzer.delimiter_detector import DelimiterDetector

            detector = DelimiterDetector(default_delimiter=self.delimiter)
            samples = [rec.data for rec in self.corpus.records]
            if samples:
                best_delim, conf, reason, ranked = detector.detect_best_delimiter(samples, fallback=self.delimiter)
                self.delimiter = best_delim
                self.delimiter_confidence = conf
                self.delimiter_selection_reason = reason
                self.delimiter_candidates = [c.to_dict() for c in ranked[:5]]
                print(f"[*] AUTO-DELIMITER DISCOVERY: Selected {repr(self.delimiter)} (confidence: {conf*100:.1f}%) | {reason}")

        # Structured crossover & field splicing (Phase 6E)
        self.enable_splicing = enable_splicing or (splicer is not None)
        self.splice_probability = float(splice_probability)
        self.max_splice_size = int(max_splice_size)
        self.splicer = (
            splicer
            if splicer is not None
            else (
                FieldSplicer(
                    delimiter=self.delimiter,
                    max_splice_size=self.max_splice_size,
                    rng=self.rng,
                )
                if self.enable_splicing
                else None
            )
        )

        # Compatibility-aware donor selection (Phase 6F)
        self.donor_policy = donor_policy.lower()
        if donor_selector is not None:
            self.donor_selector = donor_selector
            self.donor_policy = donor_selector.mode
        elif self.enable_splicing or self.donor_policy != "random":
            self.donor_selector = DonorSelector(
                mode=self.donor_policy,
                delimiter=self.delimiter,
                prefix_weight=donor_prefix_weight,
                command_weight=donor_command_weight,
                protocol_weight=donor_protocol_weight,
                extra_field_weight=donor_extra_field_weight,
                field_diff_penalty=donor_field_diff_penalty,
                depth_diff_penalty=donor_depth_diff_penalty,
                rng=self.rng,
            )
        else:
            self.donor_selector = None

        # Protocol dictionary (Phase 6A/6B)
        self.dictionary = dictionary
        if self.dictionary is None and dictionary_path is not None:
            self.dictionary = Dictionary(filepath=dictionary_path)

        # Mutation Policy (Phase 6C UCB1 & Phase 6D Contextual Bandit)
        self.mutation_policy = mutation_policy.lower()
        self.ucb_c = float(ucb_c)
        self.mutation_epsilon = float(mutation_epsilon)
        self.mutation_learning_rate = float(mutation_learning_rate)
        self.mutation_l2_reg = float(mutation_l2_reg)

        has_dict = self.dictionary is not None and len(self.dictionary) > 0
        arms = get_default_mutation_arms(
            has_dictionary=has_dict,
            boundary_aware=boundary_aware,
            include_splicing=self.enable_splicing,
        )

        # Feature extractor for contextual mutation policy (Phase 6D)
        self.mutation_feature_extractor = MutationFeatureExtractor()

        if contextual_mutation_bandit is not None:
            self.contextual_mutation_bandit = contextual_mutation_bandit
            self.mutation_policy = "contextual"
            self.mutation_bandit = None
        elif self.mutation_policy == "contextual":
            self.contextual_mutation_bandit = ContextualMutationBandit(
                operators=arms,
                feature_dim=self.mutation_feature_extractor.feature_dim,
                epsilon=self.mutation_epsilon,
                learning_rate=self.mutation_learning_rate,
                l2_reg=self.mutation_l2_reg,
                rng=self.rng,
            )
            self.mutation_bandit = None
        else:
            self.contextual_mutation_bandit = None

        if mutation_bandit is not None:
            self.mutation_bandit = mutation_bandit
            self.mutation_policy = "ucb1"
        elif self.mutation_policy == "ucb1":
            self.mutation_bandit = MutationBandit(
                operators=arms,
                exploration_constant=self.ucb_c,
                rng=self.rng,
            )
        elif self.mutation_policy != "contextual":
            self.mutation_bandit = None
            self.mutation_policy = "fixed"

        # Decoupled components
        self.mutator = Mutator(
            rng=self.rng,
            dictionary=self.dictionary,
            dictionary_probability=dictionary_probability,
            boundary_aware=boundary_aware,
            delimiter=self.delimiter,
            splicer=self.splicer,
            enable_splicing=self.enable_splicing,
            splice_probability=self.splice_probability,
            max_splice_size=self.max_splice_size,
            corpus_provider=lambda: self.corpus.records,
            donor_selector=self.donor_selector,
            donor_policy=self.donor_policy,
        )
        self.executor_type = executor_type.lower()
        if self.executor_type == "persistent":
            self.executor = PersistentExecutor(target_path=target_path, timeout=timeout)
        elif self.executor_type in ("subprocess", "default"):
            self.executor = Executor(target_path=target_path, timeout=timeout)
        else:
            raise ValueError(
                f"Unknown executor_type: '{executor_type}'. Expected 'subprocess' or 'persistent'."
            )
        self.coverage_tracker = CoverageTracker()
        self.crash_saver = CrashSaver(output_dir=crashes_dir)

        # Configurable SeedScheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif scheduler_type.lower() == "random":
            self.scheduler = RandomScheduler(rng=self.rng)
        elif scheduler_type.lower() == "heuristic":
            self.scheduler = HeuristicScheduler(epsilon=epsilon, rng=self.rng)
        elif scheduler_type.lower() in ("linear_bandit", "bandit"):
            self.scheduler = LinearBanditScheduler(
                epsilon=epsilon,
                learning_rate=learning_rate,
                l2_reg=l2_reg,
                rng=self.rng,
            )
        else:
            raise ValueError(
                f"Unknown scheduler_type: '{scheduler_type}'. Expected 'random', 'heuristic', or 'linear_bandit'."
            )

        # State-Sequence-Aware Frontier Extension (Phase 7C / 7D)
        self.enable_frontier_extension = bool(enable_frontier_extension)
        self.frontier_extension_rate = float(frontier_extension_rate)
        self.frontier_rate_mode = frontier_rate_mode.lower()
        self.min_frontier_rate = float(min_frontier_rate)
        self.max_frontier_rate = float(max_frontier_rate)
        self.frontier_decay_factor = float(frontier_decay_factor)
        self.frontier_stagnation_threshold = int(frontier_stagnation_threshold)

        if self.enable_frontier_extension:
            self.state_frontier = (
                state_frontier
                if state_frontier is not None
                else StateFrontier(
                    dictionary=self.dictionary,
                    delimiter=self.delimiter,
                    boundary_detector=self.mutator.boundary_detector,
                    rng=self.rng,
                    rate_mode=self.frontier_rate_mode,
                    frontier_rate=self.frontier_extension_rate,
                    min_frontier_rate=self.min_frontier_rate,
                    max_frontier_rate=self.max_frontier_rate,
                    decay_factor=self.frontier_decay_factor,
                    stagnation_threshold=self.frontier_stagnation_threshold,
                )
            )
        else:
            self.state_frontier = state_frontier

        self.stats = FuzzStats(
            scheduler_name=self.scheduler.name,
            executor_type=self.executor_type,
            boundary_aware=boundary_aware,
            mutation_policy=self.mutation_policy,
            ucb_c=self.ucb_c,
            mutation_epsilon=self.mutation_epsilon,
            enable_splicing=self.enable_splicing,
            donor_policy=self.donor_policy,
            enable_frontier_extension=self.enable_frontier_extension,
            frontier_rate_mode=self.frontier_rate_mode,
            current_frontier_rate=self.state_frontier.get_current_rate() if self.state_frontier else self.frontier_extension_rate,
            delimiter_mode=self.delimiter_mode,
            active_delimiter=self.delimiter.decode("latin1", errors="replace"),
            delimiter_confidence=self.delimiter_confidence,
            delimiter_selection_reason=self.delimiter_selection_reason,
            delimiter_candidates=self.delimiter_candidates,
        )

    def calibrate_corpus(self) -> None:
        """Execute initial corpus seeds to establish baseline coverage."""
        print("[*] Calibrating initial seeds to establish baseline coverage...")
        for seed_rec in self.corpus.records:
            res = self.executor.run(seed_rec.data)
            self.coverage_tracker.update(res.coverage)
            seed_rec.coverage_units = set(res.coverage)
            if self.enable_frontier_extension and self.state_frontier is not None:
                depth = extract_coverage_depth(seed_rec.coverage_units)
                self.state_frontier.register_seed(seed_rec, depth=depth, iteration=0)

        self.stats.total_coverage_units = self.coverage_tracker.total_units
        self.stats.corpus_size = len(self.corpus)
        print(f"[+] Baseline coverage: {self.stats.total_coverage_units} units from {len(self.corpus)} seeds\n")

    def run(self) -> FuzzStats:
        """Execute the coverage-guided fuzzing loop with seed scheduling.

        Returns:
            FuzzStats containing the final fuzzing statistics.
        """
        print("=" * 68)
        print(" NeuroFuzz - Phase 3: Intelligent Seed Scheduling")
        print("=" * 68)
        print(f"Target:      {self.executor.target_path}")
        print(f"Harness:     {self.executor_type.upper()}")
        print(f"Scheduler:   {self.scheduler.name.upper()} (telemetry: {self.scheduler.get_telemetry()})")
        print(f"Corpus:      {len(self.corpus)} seeds loaded")
        print(f"Iterations:  {self.iterations}")
        print(f"Timeout:     {self.timeout}s")
        print(f"Crashes Dir: {self.crash_saver.output_dir}")
        print("-" * 68)

        if self.calibrate and len(self.corpus) > 0:
            self.calibrate_corpus()

        self.stats.start_time = time.time()
        last_log_time = self.stats.start_time

        try:
            for i in range(1, self.iterations + 1):
                is_frontier_iteration = False
                active_frontier_candidate = None

                current_frontier_rate = (
                    self.state_frontier.get_current_rate()
                    if self.state_frontier is not None
                    else self.frontier_extension_rate
                )
                self.stats.current_frontier_rate = current_frontier_rate

                use_frontier = (
                    self.enable_frontier_extension
                    and self.state_frontier is not None
                    and self.state_frontier.has_candidate()
                    and (
                        self.rng.random() < current_frontier_rate
                        or self.state_frontier.is_priority_active()
                    )
                )

                if use_frontier:
                    frontier_item = self.state_frontier.next_candidate()
                    if frontier_item is not None:
                        selected_record, mutated, op_used, active_frontier_candidate = frontier_item
                        parent_depth = active_frontier_candidate.parent_depth
                        is_frontier_iteration = True
                        mutation_context = self.mutation_feature_extractor.extract(selected_record)
                    else:
                        use_frontier = False

                if not use_frontier:
                    # 1. Scheduler selects the most promising seed
                    selected_record: SeedRecord = self.scheduler.select_seed(self.corpus, iteration=i)
                    parent_depth = extract_coverage_depth(selected_record.coverage_units)

                    # Boundary telemetry inspection
                    boundaries = self.mutator.boundary_detector.find_boundaries(selected_record.data)
                    self.stats.total_detected_boundaries += len(boundaries)
                    if self.mutator.boundary_detector.has_internal_boundaries(selected_record.data):
                        self.stats.usable_boundary_seeds_count += 1

                    # Extract context features BEFORE mutation (Phase 6D - strict no-leakage)
                    mutation_context = self.mutation_feature_extractor.extract(selected_record)

                    # 2. Mutate on a copy
                    if self.mutation_policy == "contextual" and self.contextual_mutation_bandit is not None:
                        selected_op = self.contextual_mutation_bandit.select_operator(mutation_context)
                        mutated, op_used = self.mutator.mutate_with_operator(
                            selected_record.data, operator_name=selected_op
                        )
                    elif self.mutation_policy == "ucb1" and self.mutation_bandit is not None:
                        selected_op = self.mutation_bandit.select_operator()
                        mutated, op_used = self.mutator.mutate_with_operator(
                            selected_record.data, operator_name=selected_op
                        )
                    else:
                        mutated, op_used = self.mutator.mutate_with_operator(selected_record.data)
                self.stats.total_mutations += 1

                if op_used == "state_frontier_extend":
                    self.stats.frontier_extension_attempts += 1
                    self.stats.dictionary_mutations += 1
                    self.stats.dictionary_insertions += 1
                    self.stats.dictionary_insert_boundary += 1
                    self.stats.boundary_mutations += 1
                elif op_used == "dictionary_insert_boundary":
                    self.stats.dictionary_mutations += 1
                    self.stats.dictionary_insertions += 1
                    self.stats.dictionary_insert_boundary += 1
                    self.stats.boundary_mutations += 1
                elif op_used == "dictionary_insert_random":
                    self.stats.dictionary_mutations += 1
                    self.stats.dictionary_insertions += 1
                    self.stats.dictionary_insert_random += 1
                elif op_used == "dictionary_insert":
                    self.stats.dictionary_mutations += 1
                    self.stats.dictionary_insertions += 1
                    if self.boundary_aware:
                        self.stats.dictionary_insert_boundary += 1
                        self.stats.boundary_mutations += 1
                    else:
                        self.stats.dictionary_insert_random += 1
                elif op_used == "dictionary_replace":
                    self.stats.dictionary_mutations += 1
                    self.stats.dictionary_replacements += 1
                elif op_used == "field_splice":
                    self.stats.splice_mutations += 1
                else:
                    self.stats.byte_mutations += 1

                # 3. Execute target with timeout protection
                result: ExecutionResult = self.executor.run(mutated)
                self.stats.total_executions += 1

                # 4. Coverage feedback
                is_new_cov, new_units = self.coverage_tracker.update(result.coverage)
                self.stats.total_coverage_units = self.coverage_tracker.total_units

                # Track depth progression & transitions (Phase 6B)
                child_depth = extract_coverage_depth(result.coverage)
                if child_depth > parent_depth:
                    for d in range(parent_depth, child_depth):
                        trans_key = f"{d}->{d+1}"
                        self.stats.depth_transition_counts[trans_key] = (
                            self.stats.depth_transition_counts.get(trans_key, 0) + 1
                        )
                        if trans_key == "14->15" and self.splicer is not None and op_used == "field_splice":
                            self.splicer.telemetry.transition_14_15_count += 1
                    if child_depth > self.stats.deepest_state_reached:
                        self.stats.deepest_state_reached = child_depth
                        if self.splicer is not None:
                            self.splicer.telemetry.max_depth_reached = max(
                                self.splicer.telemetry.max_depth_reached, child_depth
                            )
                        print(
                            f"\n[*] DEPTH ADVANCEMENT: Depth {parent_depth} -> Depth {child_depth} at iter {i}! "
                            f"Parent: '{selected_record.name}' [{op_used}]"
                        )

                # 5. Reward definition: number of new coverage units discovered
                reward = len(new_units)
                self.scheduler.update(selected_record, reward=reward, iteration=i)
                if self.mutation_bandit is not None and op_used in self.mutation_bandit.operators:
                    self.mutation_bandit.update(op_used, reward=reward)
                if self.contextual_mutation_bandit is not None and op_used in self.contextual_mutation_bandit.operators:
                    self.contextual_mutation_bandit.update(op_used, mutation_context, reward=reward)

                if is_frontier_iteration and self.state_frontier is not None and active_frontier_candidate is not None:
                    self.state_frontier.on_execution_result(
                        active_frontier_candidate,
                        result.coverage,
                        is_new_cov,
                        new_units,
                        i,
                    )
                    if is_new_cov:
                        self.stats.frontier_extension_discoveries += 1
                        self.stats.frontier_extension_new_coverage += reward
                    if child_depth > parent_depth:
                        self.stats.frontier_extension_depth_discoveries += 1

                if is_new_cov:
                    self.stats.new_coverage_discoveries += 1
                    if op_used == "field_splice":
                        self.stats.splice_new_coverage += reward
                        if self.splicer is not None:
                            self.splicer.telemetry.coverage_discoveries += reward
                        if self.donor_selector is not None:
                            self.donor_selector.telemetry.compatible_new_coverage += reward
                            self.stats.compatible_new_coverage += reward
                    if op_used.startswith("dictionary_"):
                        self.stats.dictionary_new_coverage += reward
                    if op_used == "dictionary_insert_boundary" or (op_used == "dictionary_insert" and self.boundary_aware):
                        self.stats.boundary_new_coverage += reward

                    saved_path = self.corpus.add_interesting_input(
                        data=mutated,
                        coverage_units=new_units,
                        iteration=i,
                        save_to_disk=True,
                    )
                    self.stats.corpus_size = len(self.corpus)
                    if self.enable_frontier_extension and self.state_frontier is not None:
                        saved_record = self.corpus.records[-1]
                        self.state_frontier.register_seed(saved_record, depth=child_depth, iteration=i)
                    name_str = f" -> {saved_path.name}" if saved_path else ""
                    print(
                        f"\n[+] NEW COVERAGE: +{reward} units (total: {self.stats.total_coverage_units}) "
                        f"at iter {i}! Parent: '{selected_record.name}' [{op_used}] | "
                        f"Units: {sorted(list(new_units))} | Corpus: {len(self.corpus)}{name_str}"
                    )

                # 6. Crash detection: Save failure independently
                if result.is_crash:
                    self.stats.crashes_found += 1
                    if op_used == "field_splice":
                        self.stats.splice_crashes += 1
                    if op_used.startswith("dictionary_"):
                        self.stats.dictionary_crashes += 1
                    if op_used == "dictionary_insert_boundary" or (op_used == "dictionary_insert" and self.boundary_aware):
                        self.stats.boundary_crashes += 1

                    if result.timeout:
                        self.stats.timeouts_found += 1

                    is_new, saved_crash_path = self.crash_saver.save(
                        data=mutated,
                        result=result,
                        iteration=i,
                    )
                    if is_new and saved_crash_path is not None:
                        self.stats.unique_crashes += 1
                        print(
                            f"\n[!] NEW CRASH #{self.stats.unique_crashes} found at iter {i}! "
                            f"Type: {result.failure_type} | Code: {result.returncode} | "
                            f"Parent: '{selected_record.name}' [{op_used}] | "
                            f"Input: {repr(mutated[:20])} -> {saved_crash_path.name}"
                        )

                # Iteration callback hook (Phase 7B)
                if self.iteration_callback is not None:
                    self.iteration_callback(
                        i,
                        self.stats,
                        {
                            "iteration": i,
                            "selected_record": selected_record,
                            "mutated": mutated,
                            "op_used": op_used,
                            "result": result,
                            "is_new_cov": is_new_cov,
                            "new_units": new_units,
                            "child_depth": child_depth,
                            "parent_depth": parent_depth,
                            "donor": getattr(self.mutator, "last_donor", None),
                        },
                    )

                # Periodic progress output
                now = time.time()
                if i % self.stats_interval == 0 or i == self.iterations or (now - last_log_time >= 2.0):
                    last_log_time = now
                    print(
                        f"\r[Exec: {i:6d}/{self.iterations}] "
                        f"Rate: {self.stats.exec_per_sec:5.1f}/s | "
                        f"Sched: {self.scheduler.name} | "
                        f"Cov: {self.stats.total_coverage_units:2d} (+{self.stats.new_coverage_discoveries:2d} new) | "
                        f"AvgRwd: {self.scheduler.average_reward:.3f} | "
                        f"Corpus: {len(self.corpus):3d} | "
                        f"Crashes: {self.stats.crashes_found:2d} (Unique: {self.stats.unique_crashes:2d}) | "
                        f"Elapsed: {self.stats.elapsed_time:5.1f}s",
                        end="",
                        flush=True,
                    )

        except KeyboardInterrupt:
            print("\n\n[!] Fuzzing interrupted by user.")
        finally:
            if hasattr(self.executor, "close"):
                self.executor.close()
            self.stats.end_time = time.time()
            self.stats.corpus_size = len(self.corpus)
            self.stats.deepest_state_reached = self.coverage_tracker.max_depth
            sched_telemetry = self.scheduler.get_telemetry()
            self.stats.scheduler_telemetry = sched_telemetry
            self.stats.total_reward = sched_telemetry.get("total_reward", 0)
            self.stats.average_reward = sched_telemetry.get("average_reward", 0.0)
            self.stats.exploration_rate = sched_telemetry.get(
                "empirical_exploration_rate", sched_telemetry.get("exploration_rate", 0.0)
            )
            if self.mutation_bandit is not None:
                self.stats.mutation_bandit_telemetry = self.mutation_bandit.get_statistics()
            if self.contextual_mutation_bandit is not None:
                self.stats.contextual_mutation_bandit_telemetry = self.contextual_mutation_bandit.get_statistics(
                    feature_names=self.mutation_feature_extractor.FEATURE_NAMES
                )
            if self.splicer is not None:
                self.stats.splice_telemetry = self.splicer.get_telemetry()
            if self.donor_selector is not None:
                self.stats.donor_telemetry = self.donor_selector.get_telemetry()

        self._print_summary()
        return self.stats

    def _print_summary(self) -> None:
        """Print final summary statistics and seed performance telemetry."""
        print("\n\n" + "=" * 68)
        print(f" Fuzzing Run Summary ({self.scheduler.name.upper()} Policy)")
        print("=" * 68)
        print(f"Scheduler Strategy:        {self.scheduler.name}")
        print(f"Execution Harness:         {self.stats.executor_type}")
        print(f"Scheduler Exploration Rate:{self.stats.exploration_rate:.1%}")
        print(f"Total Executions:          {self.stats.total_executions}")
        print(f"Total Mutations:           {self.stats.total_mutations}")
        print(f"Total Coverage Units:      {self.stats.total_coverage_units}")
        if self.stats.deepest_state_reached > 0:
            print(f"Deepest State Reached:     Depth {self.stats.deepest_state_reached}")
        print(f"New Coverage Discoveries:  {self.stats.new_coverage_discoveries}")
        print(f"Total Reward (Units Found):{self.stats.total_reward}")
        print(f"Average Reward / Exec:     {self.stats.average_reward:.4f}")
        print(f"Final Corpus Size:         {self.stats.corpus_size}")
        print(f"Crashes Found:             {self.stats.crashes_found}")
        print(f"Unique Crashes:            {self.stats.unique_crashes}")
        print(f"Timeouts Encountered:      {self.stats.timeouts_found}")
        print(f"Elapsed Time:              {self.stats.elapsed_time:.2f}s")
        print(f"Average Exec Rate:         {self.stats.exec_per_sec:.1f} exec/s")

        # Protocol dictionary mutation telemetry (Phase 6A)
        if self.stats.dictionary_mutations > 0 or (self.mutator.dictionary and len(self.mutator.dictionary) > 0):
            print("\n--- Protocol Dictionary Mutation Telemetry (Phase 6A) ---")
            print(f"Dictionary Loaded Tokens:  {len(self.mutator.dictionary) if self.mutator.dictionary else 0}")
            print(f"Configured Probability:    {self.mutator.dictionary_probability:.1%}")
            print(f"Byte Mutations:            {self.stats.byte_mutations}")
            print(f"Dictionary Mutations:      {self.stats.dictionary_mutations} ({self.stats.dictionary_mutation_rate:.1%})")
            print(f"  - Insertions:            {self.stats.dictionary_insertions}")
            print(f"  - Replacements:          {self.stats.dictionary_replacements}")
            print(f"Dict New Cov Discoveries:  {self.stats.dictionary_new_coverage} units")
            print(f"Dict Crashes Discovered:   {self.stats.dictionary_crashes}")

        # Boundary-Aware Dictionary Mutation Telemetry (Phase 6B)
        if self.boundary_aware or self.stats.boundary_mutations > 0:
            print("\n--- Boundary-Aware Mutation Telemetry (Phase 6B) ---")
            print(f"Boundary Awareness Mode:   {'ENABLED' if self.boundary_aware else 'DISABLED'}")
            print(f"Boundary Mutator Type:     BoundaryDetector(delimiter=b'|')")
            print(f"Boundary Insertions:       {self.stats.dictionary_insert_boundary}")
            print(f"Random Insertions:         {self.stats.dictionary_insert_random}")
            print(f"Boundary Mutation Share:   {self.stats.boundary_mutation_rate:.1%} of dict mutations")
            print(f"Boundary Cov Discoveries:  {self.stats.boundary_new_coverage} units")
            print(f"Boundary Crashes Found:    {self.stats.boundary_crashes}")
            print(f"Total Detected Boundaries: {self.stats.total_detected_boundaries}")
            print(f"Usable Boundary Seeds:     {self.stats.usable_boundary_seeds_count}")

        # UCB1 Learned Mutation Operator Telemetry (Phase 6C)
        if self.mutation_bandit is not None:
            bandit_stats = self.mutation_bandit.get_statistics()
            total_n = bandit_stats.get("total_selections", 0)
            explorations = bandit_stats.get("explorations", 0)
            exploitations = bandit_stats.get("exploitations", 0)
            print("\n--- UCB1 Learned Mutation Operator Telemetry (Phase 6C) ---")
            print(f"Exploration Constant (c):  {bandit_stats.get('exploration_constant', 1.0):.2f}")
            print(f"Total Operator Pulls (N):  {total_n}")
            print(f"Explorations:              {explorations} ({explorations / max(1, total_n):.1%})")
            print(f"Exploitations:             {exploitations} ({exploitations / max(1, total_n):.1%})")
            print(f"\n{'Operator Name':<28} {'Selected':<10} {'Share':<8} {'Mean Rew':<10} {'UCB Score':<12} {'Nonzero':<9} {'Cov Units':<10}")
            print("-" * 92)
            ops_dict = bandit_stats.get("operators", {})
            sorted_ops = sorted(ops_dict.items(), key=lambda kv: kv[1]["selections"], reverse=True)
            for op_name, op_info in sorted_ops:
                score_str = f"{op_info['ucb_score']:.6f}" if op_info["ucb_score"] != float("inf") else "+inf"
                print(
                    f"{op_name:<28} "
                    f"{op_info['selections']:<10} "
                    f"{op_info['selection_rate']:<8.1%} "
                    f"{op_info['mean_reward']:<10.4f} "
                    f"{score_str:<12} "
                    f"{op_info['nonzero_rewards']:<9} "
                    f"{op_info['coverage_discoveries']:<10}"
                )

        # Contextual Learned Mutation Operator Telemetry (Phase 6D)
        if self.contextual_mutation_bandit is not None:
            c_stats = self.contextual_mutation_bandit.get_statistics(
                feature_names=self.mutation_feature_extractor.FEATURE_NAMES
            )
            total_n = c_stats.get("total_selections", 0)
            explorations = c_stats.get("explorations", 0)
            exploitations = c_stats.get("exploitations", 0)
            print("\n--- Contextual Learned Mutation Operator Telemetry (Phase 6D) ---")
            print(f"Policy Strategy:           Contextual Linear Bandit (epsilon-greedy)")
            print(f"Exploration Rate (eps):    {c_stats.get('epsilon', 0.2):.2f}")
            print(f"Learning Rate (eta):       {c_stats.get('learning_rate', 0.05):.4f}")
            print(f"L2 Regularization (lambda):{c_stats.get('l2_reg', 0.001):.4f}")
            print(f"Total Operator Pulls (N):  {total_n}")
            print(f"Explorations:              {explorations} ({explorations / max(1, total_n):.1%})")
            print(f"Exploitations:             {exploitations} ({exploitations / max(1, total_n):.1%})")
            print(f"\n{'Operator Name':<28} {'Selected':<10} {'Share':<8} {'Mean Rew':<10} {'Nonzero':<9} {'Cov Units':<10} {'Model MSE':<12}")
            print("-" * 92)
            ops_dict = c_stats.get("operators", {})
            sorted_ops = sorted(ops_dict.items(), key=lambda kv: kv[1]["selections"], reverse=True)
            for op_name, op_info in sorted_ops:
                print(
                    f"{op_name:<28} "
                    f"{op_info['selections']:<10} "
                    f"{op_info['selection_rate']:<8.1%} "
                    f"{op_info['mean_reward']:<10.4f} "
                    f"{op_info['nonzero_rewards']:<9} "
                    f"{op_info['coverage_discoveries']:<10} "
                    f"{op_info['model_mse']:<12.6f}"
                )

            # Per-Arm Learned Contextual Weights Table
            print("\n--- Per-Arm Learned Contextual Weights (Online SGD) ---")
            feat_names = self.mutation_feature_extractor.FEATURE_NAMES
            header = f"{'Operator':<26} " + " ".join(f"{fn:>10}" for fn in feat_names)
            print(header)
            print("-" * len(header))
            for op_name, op_info in sorted_ops:
                w_dict = op_info.get("weights", {})
                w_str = " ".join(f"{w_dict.get(fn, 0.0):>+10.4f}" for fn in feat_names)
                print(f"{op_name:<26} {w_str}")

        # Structured Crossover & Field Splicing Telemetry (Phase 6E)
        if self.enable_splicing or self.stats.splice_mutations > 0:
            sp_stats = self.stats.splice_telemetry if self.stats.splice_telemetry else (
                self.splicer.get_telemetry() if self.splicer else {}
            )
            print("\n--- Structured Field Splicing Telemetry (Phase 6E) ---")
            print(f"Splicing Mode:             {'ENABLED' if self.enable_splicing else 'DISABLED'}")
            print(f"Splice Probability:        {self.splice_probability:.1%}")
            print(f"Max Splice Size:           {self.max_splice_size} bytes")
            print(f"Total Splice Mutations:    {self.stats.splice_mutations}")
            print(f"Splice Attempts:           {sp_stats.get('splice_attempts', 0)}")
            print(f"Successful Splices:        {sp_stats.get('successful_splices', 0)} ({sp_stats.get('success_rate', 0.0):.1%})")
            print(f"Rejected Splices:          {sp_stats.get('rejected_splices', 0)}")
            print(f"Duplicate Prevention Evts: {sp_stats.get('duplicate_prevention_events', 0)}")
            print(f"Splice New Cov Discoveries:{self.stats.splice_new_coverage} units")
            print(f"Splice Crashes Found:      {self.stats.splice_crashes}")
            print(f"14->15 Transitions via Splice: {sp_stats.get('transition_14_15_count', 0)}")
            print(f"Avg Fields Parent A:       {sp_stats.get('avg_parent_a_fields', 0.0):.2f}")
            print(f"Avg Fields Parent B:       {sp_stats.get('avg_parent_b_fields', 0.0):.2f}")
            print(f"Avg Fields Result:         {sp_stats.get('avg_result_fields', 0.0):.2f}")
            strat_counts = sp_stats.get("strategy_counts", {})
            print("Strategy Breakdown:")
            for s_name, s_cnt in sorted(strat_counts.items()):
                print(f"  - {s_name:<24}: {s_cnt}")

        # Compatibility-Aware Donor Selection Telemetry (Phase 6F)
        if self.donor_selector is not None or self.stats.donor_policy != "random" or self.stats.donor_telemetry:
            d_stats = self.stats.donor_telemetry if self.stats.donor_telemetry else (
                self.donor_selector.get_telemetry() if self.donor_selector else {}
            )
            if d_stats.get("total_attempts", 0) > 0:
                print("\n--- Compatibility-Aware Donor Selection Telemetry (Phase 6F) ---")
                print(f"Donor Selection Policy:    {d_stats.get('policy_mode', self.donor_policy).upper()}")
                print(f"Donor Selection Attempts:  {d_stats.get('total_attempts', 0)}")
                print(f"Successful Donor Selects:  {d_stats.get('successful_selections', 0)}")
                print(f"Single-Seed Fallbacks:     {d_stats.get('single_seed_fallbacks', 0)}")
                print(f"Average Compatibility Score:{d_stats.get('average_compatibility_score', 0.0):.3f}")
                print(f"Average Common Prefix Len: {d_stats.get('average_common_prefix_length', 0.0):.2f} fields")
                print(f"Same-Command Pairing Rate: {d_stats.get('same_command_pairing_rate', 0.0):.1%}")
                print(f"Same-Protocol Pairing Rate:{d_stats.get('same_protocol_pairing_rate', 0.0):.1%}")
                print(f"Useful Donor Rate:         {d_stats.get('useful_donor_rate', 0.0):.1%} (donors with extra fields)")
                print(f"Average Extra Fields:      {d_stats.get('average_donor_extra_fields', 0.0):.2f}")
                print(f"DIAG-DIAG Pairings:        {d_stats.get('diag_diag_pairings', 0)}")
        # State-Frontier Extension Telemetry (Phase 7C)
        if self.enable_frontier_extension and self.state_frontier is not None:
            ft = self.state_frontier.telemetry
            telemetry_dict = ft.to_dict()
            telemetry_dict["frontier_mutation_share"] = self.stats.frontier_extension_attempts / max(1, self.stats.total_mutations)
            self.stats.frontier_telemetry = telemetry_dict
            print("\n--- State-Frontier Extension Telemetry (Phase 7C) ---")
            print(f"Frontier Mode:             ENABLED (rate={self.frontier_extension_rate*100:.1f}%)")
            print(f"Frontier Extension Attempts:{self.stats.frontier_extension_attempts}")
            print(f"Unique Candidates Created: {ft.unique_frontier_candidates}")
            print(f"Candidates Exhausted:      {ft.frontier_candidates_exhausted}")
            print(f"Frontier Cov Discoveries:  {self.stats.frontier_extension_discoveries} units")
            print(f"Frontier Depth Discoveries:{self.stats.frontier_extension_depth_discoveries}")
            print(f"Deepest Frontier Reached:  Depth {self.state_frontier.max_depth}")
            print(f"Successful Frontier Rate:  {ft.successful_frontier_mutation_rate*100:.2f}%")
            print(f"Avg Candidate Gen Cost:    {ft.candidate_generation_cost_ms:.3f} ms")
            print(f"Frontier Mutation Share:   {telemetry_dict['frontier_mutation_share']*100:.1f}%")

        # Automated Delimiter Discovery Telemetry (Phase 7D)
        print("\n--- Protocol Delimiter Telemetry (Phase 7D) ---")
        print(f"Delimiter Discovery Mode:  {self.stats.delimiter_mode.upper()}")
        print(f"Active Delimiter:          {repr(self.stats.active_delimiter)}")
        print(f"Delimiter Confidence:      {self.stats.delimiter_confidence*100:.1f}%")
        print(f"Selection Reason:          {self.stats.delimiter_selection_reason}")
        if self.stats.delimiter_candidates:
            print("Ranked Candidates (Top 3):")
            for cand in self.stats.delimiter_candidates[:3]:
                print(f"  - Delim: {repr(cand['delimiter']):<6} Conf: {cand['confidence']*100:.1f}% | {cand['selection_reason']}")

        # Adaptive Frontier Rate Telemetry (Phase 7D)
        if self.enable_frontier_extension and self.state_frontier is not None:
            ft = self.state_frontier.telemetry
            print("\n--- Adaptive Frontier Rate Telemetry (Phase 7D) ---")
            print(f"Rate Allocation Mode:      {ft.rate_mode.upper()}")
            print(f"Current Frontier Rate:     {ft.current_frontier_rate*100:.2f}% (nominal: {ft.starting_frontier_rate*100:.1f}%, bounds: [{ft.min_frontier_rate*100:.1f}%, {ft.max_frontier_rate*100:.1f}%])")
            print(f"Rate Adjustments:          {ft.rate_changes_count}")
            print(f"Last Adjustment Reason:    {ft.last_rate_change_reason} (iter {ft.last_rate_change_iteration})")
            print(f"Stagnant Attempts Window:  {ft.stagnant_frontier_attempts}")
            print(f"Iterations Since Advance:  {ft.iterations_since_last_depth_advance}")

        # Depth Transitions Telemetry (Structured Protocol Benchmark)
        if self.stats.depth_transition_counts:
            print("\n--- Protocol Depth Transitions Telemetry ---")
            print(f"{'Transition':<16} {'Occurrences':<14}")
            print("-" * 30)
            for trans_name, cnt in sorted(self.stats.depth_transition_counts.items()):
                print(f"{trans_name:<16} {cnt:<14}")
            print(f"Deepest Depth Reached:     {self.stats.deepest_state_reached}")

        # Contextual bandit learned weights table
        if isinstance(self.scheduler, LinearBanditScheduler):
            telemetry = self.scheduler.get_telemetry()
            print("\n--- Learned Contextual Bandit Weights (Online SGD) ---")
            print(f"{'Feature Name':<24} {'Weight (w_i)':<14} {'Direction':<12}")
            print("-" * 68)
            weights = telemetry.get("learned_weights", {})
            for feat_name, w_val in weights.items():
                direction = "Positive" if w_val > 0.001 else ("Negative" if w_val < -0.001 else "Neutral")
                print(f"{feat_name:<24} {w_val:>+12.6f}  {direction:<12}")
            print(f"Model Cumulative Updates:   {telemetry.get('model_updates', 0)}")
            print(f"Model Mean Squared Error:   {telemetry.get('model_mse', 0.0):.6f}")
            print(f"Model Mean Absolute Error:  {telemetry.get('model_mae', 0.0):.6f}")

        # Top productive seeds table
        productive_seeds = sorted(
            self.corpus.records,
            key=lambda s: (s.total_coverage_discovered, s.times_produced_coverage),
            reverse=True,
        )
        print("\n--- Seed Performance Breakdown (Top 5) ---")
        print(f"{'Seed Name':<32} {'Selected':<10} {'Cov Found':<10} {'Yield Ratio':<12}")
        print("-" * 68)
        for s in productive_seeds[:5]:
            yield_ratio = s.total_coverage_discovered / max(1, s.times_selected)
            print(f"{s.name[:30]:<32} {s.times_selected:<10} {s.total_coverage_discovered:<10} {yield_ratio:<12.3f}")
        print("=" * 68)


def main() -> None:
    """CLI entrypoint for running the coverage-guided fuzzer with seed scheduling."""
    parser = argparse.ArgumentParser(
        description="NeuroFuzz - Contextual Bandit Coverage-Guided Fuzzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to the target executable to fuzz.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["random", "heuristic", "linear_bandit"],
        default="random",
        help="Seed scheduling strategy to deploy (default: random).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Exploration probability for bandit/heuristic scheduler.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (eta) for online linear model updates.",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.001,
        help="L2 regularization coefficient for online linear model.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=None,
        help="Path to protocol dictionary file containing valid tokens (Phase 6A).",
    )
    parser.add_argument(
        "--dictionary-probability",
        type=float,
        default=0.0,
        help="Probability of selecting a dictionary mutation over byte mutations (0.0 to 1.0).",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus",
        help="Path to initial seed corpus directory.",
    )
    parser.add_argument(
        "--crashes",
        type=str,
        default="crashes",
        help="Path to directory where crashes will be saved.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations to run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-execution timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Statistics display interval (in iterations).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable pre-run initial seed calibration.",
    )
    parser.add_argument(
        "--boundary-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable boundary-aware dictionary insertions (Phase 6B, default: True).",
    )
    parser.add_argument(
        "--mutation-policy",
        type=str,
        choices=["fixed", "ucb1", "contextual"],
        default="contextual",
        help="Mutation operator selection policy: 'fixed', 'ucb1', or 'contextual' (Phase 6D, default: contextual).",
    )
    parser.add_argument(
        "--ucb-c",
        type=float,
        default=1.0,
        help="Exploration parameter c for UCB1 mutation bandit (default: 1.0).",
    )
    parser.add_argument(
        "--mutation-epsilon",
        type=float,
        default=0.20,
        help="Exploration probability for contextual mutation bandit (default: 0.20).",
    )
    parser.add_argument(
        "--mutation-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for contextual mutation bandit online models (default: 0.05).",
    )
    parser.add_argument(
        "--mutation-l2-reg",
        type=float,
        default=0.001,
        help="L2 regularization for contextual mutation bandit online models (default: 0.001).",
    )
    parser.add_argument(
        "--enable-splicing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structured field splicing / crossover (Phase 6E, default: True).",
    )
    parser.add_argument(
        "--splice-probability",
        type=float,
        default=0.0,
        help="Probability of selecting field_splice in fixed mutation policy (default: 0.0).",
    )
    parser.add_argument(
        "--max-splice-size",
        type=int,
        default=512,
        help="Maximum allowed byte length for spliced candidates (default: 512).",
    )
    parser.add_argument(
        "--donor-policy",
        type=str,
        choices=["random", "compatible"],
        default="compatible",
        help="Donor parent selection policy for structured splicing: 'random' or 'compatible' (Phase 6F, default: compatible).",
    )
    parser.add_argument(
        "--donor-prefix-weight",
        type=float,
        default=3.0,
        help="Compatibility weight for matching field prefix length (default: 3.0).",
    )
    parser.add_argument(
        "--donor-command-weight",
        type=float,
        default=2.0,
        help="Compatibility bonus for identical command field (default: 2.0).",
    )
    parser.add_argument(
        "--donor-protocol-weight",
        type=float,
        default=1.0,
        help="Compatibility bonus for identical protocol header (default: 1.0).",
    )
    parser.add_argument(
        "--donor-extra-field-weight",
        type=float,
        default=1.0,
        help="Compatibility weight for donor having additional fields beyond prefix (default: 1.0).",
    )
    parser.add_argument(
        "--donor-field-diff-penalty",
        type=float,
        default=0.5,
        help="Compatibility penalty per field count difference (default: 0.5).",
    )
    parser.add_argument(
        "--donor-depth-diff-penalty",
        type=float,
        default=0.25,
        help="Compatibility penalty per coverage depth difference (default: 0.25).",
    )
    parser.add_argument(
        "--executor",
        choices=["subprocess", "persistent"],
        default="persistent",
        help="Target execution harness ('subprocess' or 'persistent', default: 'persistent').",
    )
    parser.add_argument(
        "--frontier-extension",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable state-sequence-aware frontier mutation (Phase 7C, default: True).",
    )
    parser.add_argument(
        "--frontier-extension-rate",
        type=float,
        default=0.20,
        help="Probability of selecting frontier extension candidate when available (default: 0.20).",
    )
    parser.add_argument(
        "--frontier-rate-mode",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Frontier allocation rate mode: 'fixed' or 'adaptive' (Phase 7D, default: adaptive).",
    )
    parser.add_argument(
        "--min-frontier-rate",
        type=float,
        default=0.02,
        help="Minimum frontier allocation rate floor in adaptive mode (default: 0.02).",
    )
    parser.add_argument(
        "--max-frontier-rate",
        type=float,
        default=0.30,
        help="Maximum frontier allocation rate ceiling in adaptive mode (default: 0.30).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="Protocol field delimiter character (default: '|').",
    )
    parser.add_argument(
        "--delimiter-mode",
        choices=["static", "auto"],
        default="auto",
        help="Delimiter discovery mode: 'static' or 'auto' (Phase 7D runtime inference, default: auto).",
    )

    args = parser.parse_args()

    # Automatically resolve .exe extension on Windows if omitted
    target_path = Path(args.target)
    if not target_path.exists() and sys.platform.startswith("win"):
        candidate = target_path.with_suffix(".exe")
        if candidate.exists():
            target_path = candidate

    fuzzer = Fuzzer(
        target_path=target_path,
        corpus_dir=args.corpus,
        crashes_dir=args.crashes,
        iterations=args.iterations,
        timeout=args.timeout,
        seed=args.seed,
        stats_interval=args.interval,
        calibrate=not args.no_calibrate,
        scheduler_type=args.scheduler,
        epsilon=args.epsilon,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        dictionary_path=args.dictionary,
        dictionary_probability=args.dictionary_probability,
        boundary_aware=args.boundary_aware,
        mutation_policy=args.mutation_policy,
        ucb_c=args.ucb_c,
        mutation_epsilon=args.mutation_epsilon,
        mutation_learning_rate=args.mutation_learning_rate,
        mutation_l2_reg=args.mutation_l2_reg,
        enable_splicing=args.enable_splicing,
        splice_probability=args.splice_probability,
        max_splice_size=args.max_splice_size,
        donor_policy=args.donor_policy,
        donor_prefix_weight=args.donor_prefix_weight,
        donor_command_weight=args.donor_command_weight,
        donor_protocol_weight=args.donor_protocol_weight,
        donor_extra_field_weight=args.donor_extra_field_weight,
        donor_field_diff_penalty=args.donor_field_diff_penalty,
        donor_depth_diff_penalty=args.donor_depth_diff_penalty,
        executor_type=args.executor,
        enable_frontier_extension=args.frontier_extension,
        frontier_extension_rate=args.frontier_extension_rate,
        frontier_rate_mode=args.frontier_rate_mode,
        min_frontier_rate=args.min_frontier_rate,
        max_frontier_rate=args.max_frontier_rate,
        delimiter=args.delimiter,
        delimiter_mode=args.delimiter_mode,
    )
    fuzzer.run()


if __name__ == "__main__":
    main()
