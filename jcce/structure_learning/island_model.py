"""
Hierarchical Island Model for Processor Evolution.

Structure:
- Level 1: 5 processor-type islands (MLP, Transformer, GNN, ELM, Mamba)
- Level 2: 2 GPU shards per processor island
- Total: 10 sub-populations

Migration:
- Intra-processor: Full genome transfer (same architecture)
- Inter-processor: Structure-only transfer (A matrix, hyperparams)

Key insight: Fast processors (ELM) scout the structure space,
then share good structures with slow processors (Transformer/Mamba).

Usage:
    from island_model import HierarchicalIslandModel

    model = HierarchicalIslandModel(
        processor_types=['elm', 'mlp', 'transformer'],
        gpu_ids=[0, 1],
        population_per_island=8
    )

    for gen in range(n_generations):
        results = model.run_generation(X_train, Y_train, gen)
        pareto = model.get_global_pareto_front()
"""

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


@dataclass
class IslandConfig:
    """Configuration for a single island."""

    processor_type: str
    gpu_id: int
    population_size: int
    golem_iterations: int  # Processor-specific budget


@dataclass
class Individual:
    """Individual in the population."""

    A_topology: np.ndarray  # Adjacency matrix
    lambda_1: float
    lambda_2: float
    lr: float
    processor_type: str
    fitness: float = 0.0
    accuracy: float = 0.0
    sparsity: float = 0.0
    h_A: float = float("inf")
    effect_loss: float = 1.0
    processor_params: Optional[Any] = None
    generation: int = 0
    causal_effects: Optional[Dict[str, float]] = None

    def copy(self) -> "Individual":
        """Create a copy of this individual."""
        return Individual(
            A_topology=self.A_topology.copy(),
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            lr=self.lr,
            processor_type=self.processor_type,
            fitness=self.fitness,
            accuracy=self.accuracy,
            sparsity=self.sparsity,
            h_A=self.h_A,
            effect_loss=self.effect_loss,
            processor_params=self.processor_params,
            generation=self.generation,
            causal_effects=self.causal_effects.copy() if self.causal_effects else None,
        )


class Island:
    """
    Single island (one processor type on one GPU).

    Manages a sub-population of individuals, all using the same
    processor type and running on the same GPU.
    """

    # Hyperparameter search spaces
    LAMBDA_1_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    LAMBDA_2_OPTIONS = [0.01, 0.1, 1.0, 10.0, 100.0]
    LR_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    def __init__(self, config: IslandConfig, n_vars: int, seed: int = 42):
        """
        Initialize island.

        Args:
            config: Island configuration
            n_vars: Number of variables in the causal graph
            seed: Random seed
        """
        self.config = config
        self.n_vars = n_vars
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.generation = 0

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize random population."""
        # A_topology needs to be (n_vars+1, n_vars+1) to include Y
        # This matches GOLEM's expectation: n_total = n_vars + 1
        n_total = self.n_vars + 1

        for i in range(self.config.population_size):
            # Random sparse adjacency matrix (includes Y as last variable)
            A = self.rng.randn(n_total, n_total) * 0.1
            np.fill_diagonal(A, 0)  # No self-loops

            # Y (last row) has no outgoing edges (outcome sink constraint)
            A[self.n_vars, :] = 0.0

            # Make it sparse (keep ~20% of edges)
            mask = self.rng.rand(n_total, n_total) < 0.2
            A = A * mask

            individual = Individual(
                A_topology=A,
                lambda_1=self.rng.choice(self.LAMBDA_1_OPTIONS),
                lambda_2=self.rng.choice(self.LAMBDA_2_OPTIONS),
                lr=self.rng.choice(self.LR_OPTIONS),
                processor_type=self.config.processor_type,
                generation=0,
            )
            self.population.append(individual)

    def evaluate(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        max_iter: int = None,
        verbose: bool = False,
        use_spectral_constraint: bool = True,
        enable_pruning: bool = True,
        use_v7: bool = False,
        Y_continuous: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all individuals in the population.

        Args:
            X_train: Training features
            Y_train: Training labels (binary for classification)
            max_iter: Max GOLEM iterations (default: config.golem_iterations)
            verbose: Print progress
            use_spectral_constraint: Use O(d²) spectral DAG constraint
            enable_pruning: Enable dynamic edge pruning
            use_v7: Enable effect estimation with bi-directed edges
            Y_continuous: Continuous Y for effect estimation (optional)

        Returns:
            Dictionary with evaluation statistics
        """
        from jcce.structure_learning.jcce_learner import (
            _learn_structure_legacy,
            create_processor,
        )

        if use_v7:
            from jcce.structure_learning.jcce_learner import learn_structure

        if max_iter is None:
            max_iter = self.config.golem_iterations

        n_samples, n_features = X_train.shape
        Y_idx = n_features  # Target is last variable (index in augmented space)

        # Augment data with Y (only needed for v4, v7 handles internally)
        X_aug = np.concatenate([X_train, Y_train.reshape(-1, 1)], axis=1)

        results = {
            "evaluated": 0,
            "best_accuracy": 0.0,
            "best_fitness": 0.0,
            "mean_accuracy": 0.0,
        }

        for i, ind in enumerate(self.population):
            try:
                # Get JAX device - support both GPU and CPU backends
                backend = jax.default_backend()
                if backend == "gpu":
                    device = jax.devices("gpu")[self.config.gpu_id]
                else:
                    device = jax.devices("cpu")[0]

                with jax.default_device(device):
                    # Create processor
                    key = random.PRNGKey(self.seed + self.generation * 1000 + i)
                    processor = create_processor(self.config.processor_type, key)

                    # Run GOLEM with individual's hyperparameters
                    if use_v7:
                        # Use effect estimation version
                        # Expects X only (not augmented); Y handled separately
                        A_est, _, params, metrics = learn_structure(
                            data=jnp.array(X_train),  # Pass X only, not X_aug
                            Y=jnp.array(Y_train),
                            Y_idx=Y_idx,
                            processor=processor,
                            key=key,
                            processor_type=self.config.processor_type,
                            lambda_1=ind.lambda_1,
                            lambda_2_init=ind.lambda_2,
                            lr=ind.lr,
                            max_iter=max_iter,
                            patience=25,
                            verbose=False,
                            A_init=jnp.array(ind.A_topology),
                            # Effect parameters (use defaults)
                            effect_hidden_dim=128,
                            effect_embed_dim=32,
                            lambda_effect=5.0,
                            effect_warmup_iter=20,
                            lambda_confound_sparse=0.001,
                            lambda_bow=0.01,  # Function uses lambda_bow
                            # Continuous Y for effect estimation
                            Y_continuous=jnp.array(Y_continuous)
                            if Y_continuous is not None
                            else None,
                            # Post-hoc effect refinement
                            effect_refinement_iters=50,
                        )
                    else:
                        # Standard optimization (expects X_aug with Y included)
                        A_est, _, params, metrics = _learn_structure_legacy(
                            data=jnp.array(X_aug),  # Augmented data (X with Y appended)
                            Y=jnp.array(Y_train),
                            Y_idx=Y_idx,
                            processor=processor,
                            key=key,
                            processor_type=self.config.processor_type,
                            lambda_1=ind.lambda_1,
                            lambda_2_init=ind.lambda_2,
                            lr=ind.lr,
                            max_iter=max_iter,
                            patience=10,
                            verbose=False,
                            A_init=jnp.array(ind.A_topology),
                            use_spectral_constraint=use_spectral_constraint,
                            enable_pruning=enable_pruning,
                        )

                    # Extract metrics
                    # Compute classification accuracy via cross-validation
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import cross_val_score

                    # Extract Markov Blanket features
                    A_np = np.array(A_est)
                    mb_indices = np.where(np.abs(A_np[:, Y_idx]) > 0.01)[0]
                    mb_indices = mb_indices[mb_indices < n_features]

                    if len(mb_indices) > 0:
                        X_mb = X_train[:, mb_indices]
                    else:
                        # Fallback to top 3 features by A weights
                        weights = np.abs(A_np[:n_features, Y_idx])
                        top_k = min(3, n_features)
                        mb_indices = np.argsort(weights)[-top_k:]
                        X_mb = X_train[:, mb_indices]

                    clf = LogisticRegression(max_iter=500, random_state=42)
                    try:
                        cv_scores = cross_val_score(clf, X_mb, Y_train, cv=3, scoring="accuracy")
                        accuracy = float(np.mean(cv_scores))
                    except:
                        accuracy = 0.5

                    # Compute sparsity (A is n_total x n_total where n_total = n_vars + 1)
                    n_edges = int(np.sum(np.abs(A_np) > 0.01))
                    n_total = self.n_vars + 1
                    max_edges = n_total * (n_total - 1)  # Exclude diagonal
                    sparsity = 1.0 - n_edges / max_edges if max_edges > 0 else 0.0

                    # Get h_A from metrics
                    h_A = metrics.get("final_h_A", float("inf"))

                    # Update individual
                    ind.A_topology = A_np
                    ind.accuracy = accuracy
                    ind.sparsity = sparsity
                    ind.h_A = h_A
                    ind.processor_params = params
                    ind.generation = self.generation

                    # Extract effect metrics if available
                    if use_v7:
                        ind.effect_loss = metrics.get("effect_loss", 1.0)
                        ind.causal_effects = metrics.get("causal_effects", {})
                        # 4-objective fitness: include effect_loss
                        ind.fitness = (
                            0.5 * accuracy + 0.2 * sparsity + 0.3 * (1.0 - ind.effect_loss)
                        )
                    else:
                        ind.fitness = 0.7 * accuracy + 0.3 * sparsity

                    results["evaluated"] += 1

            except Exception as e:
                if verbose:
                    backend_name = jax.default_backend().upper()
                    print(
                        f"  Island {self.config.processor_type} {backend_name} {self.config.gpu_id}: "
                        f"Individual {i} failed: {e}"
                    )
                # Assign poor fitness to failed individuals
                ind.fitness = 0.0
                ind.accuracy = 0.0

            # Cleanup
            jax.clear_caches()
            gc.collect()

        # Update statistics
        accuracies = [ind.accuracy for ind in self.population]
        fitnesses = [ind.fitness for ind in self.population]

        results["best_accuracy"] = max(accuracies) if accuracies else 0.0
        results["best_fitness"] = max(fitnesses) if fitnesses else 0.0
        results["mean_accuracy"] = np.mean(accuracies) if accuracies else 0.0

        # Update Pareto front
        self._update_pareto_front()

        return results

    def _update_pareto_front(self):
        """Update Pareto front from current population."""
        self.pareto_front = []

        for ind in self.population:
            dominated = False
            for other in self.population:
                if other is not ind and self._dominates(other, ind):
                    dominated = True
                    break
            if not dominated:
                self.pareto_front.append(ind.copy())

    @staticmethod
    def _dominates(a: Individual, b: Individual) -> bool:
        """Check if individual a dominates individual b."""
        better_in_all = a.accuracy >= b.accuracy and a.sparsity >= b.sparsity and a.h_A <= b.h_A
        strictly_better = a.accuracy > b.accuracy or a.sparsity > b.sparsity or a.h_A < b.h_A
        return better_in_all and strictly_better

    def select_and_reproduce(self):
        """
        Selection and reproduction (crossover + mutation).

        Uses tournament selection and produces offspring.
        """
        new_population = []

        # Elitism: keep top 2
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend([ind.copy() for ind in sorted_pop[:2]])

        # Generate rest via crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(k=3)
            parent2 = self._tournament_select(k=3)

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        self.population = new_population[: self.config.population_size]
        self.generation += 1

    def _tournament_select(self, k: int = 3) -> Individual:
        """Tournament selection."""
        candidates = self.rng.choice(
            self.population, size=min(k, len(self.population)), replace=False
        )
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover two parents to create child."""
        # Blend adjacency matrices
        alpha = self.rng.rand()
        A_child = alpha * parent1.A_topology + (1 - alpha) * parent2.A_topology

        # Random hyperparameter selection
        child = Individual(
            A_topology=A_child,
            lambda_1=parent1.lambda_1 if self.rng.rand() < 0.5 else parent2.lambda_1,
            lambda_2=parent1.lambda_2 if self.rng.rand() < 0.5 else parent2.lambda_2,
            lr=parent1.lr if self.rng.rand() < 0.5 else parent2.lr,
            processor_type=self.config.processor_type,
            generation=self.generation,
        )
        return child

    def _mutate(self, ind: Individual, mutation_rate: float = 0.1) -> Individual:
        """Mutate individual."""
        # Mutate adjacency matrix
        if self.rng.rand() < mutation_rate:
            noise = self.rng.randn(*ind.A_topology.shape) * 0.05
            ind.A_topology = ind.A_topology + noise
            np.fill_diagonal(ind.A_topology, 0)

        # Mutate hyperparameters
        if self.rng.rand() < mutation_rate:
            ind.lambda_1 = self.rng.choice(self.LAMBDA_1_OPTIONS)
        if self.rng.rand() < mutation_rate:
            ind.lambda_2 = self.rng.choice(self.LAMBDA_2_OPTIONS)
        if self.rng.rand() < mutation_rate:
            ind.lr = self.rng.choice(self.LR_OPTIONS)

        return ind

    def get_elites(self, n: int = 2) -> List[Individual]:
        """Get top n individuals by fitness."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [ind.copy() for ind in sorted_pop[:n]]

    def get_best(self) -> Optional[Individual]:
        """Get best individual."""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness).copy()

    def get_pareto_front(self) -> List[Individual]:
        """Get current Pareto front."""
        return [ind.copy() for ind in self.pareto_front]

    def accept_migrants(self, migrants: List[Individual], replace_worst: bool = True):
        """Accept migrant individuals into population."""
        if replace_worst:
            # Replace worst individuals
            self.population.sort(key=lambda x: x.fitness)
            for i, migrant in enumerate(migrants):
                if i < len(self.population):
                    # Update processor type for compatibility
                    migrant.processor_type = self.config.processor_type
                    self.population[i] = migrant

    def inject_migrant(self, migrant: Individual):
        """Inject a single migrant, replacing worst individual."""
        if self.population:
            worst_idx = min(range(len(self.population)), key=lambda i: self.population[i].fitness)
            migrant.processor_type = self.config.processor_type
            self.population[worst_idx] = migrant

    def create_from_structure(
        self, A_topology: np.ndarray, lambda_1: float, lambda_2: float, lr: float
    ) -> Individual:
        """Create new individual from structure (inter-processor migration)."""
        return Individual(
            A_topology=A_topology.copy(),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lr=lr,
            processor_type=self.config.processor_type,
            generation=self.generation,
        )


class HierarchicalIslandModel:
    """
    Hierarchical island model for unified processor evolution.

    Key insight: Fast processors (ELM) scout the structure space,
    then share good structures with slow processors (Transformer/Mamba).

    Structure:
    - Level 1: Multiple processor-type islands
    - Level 2: GPU shards per processor island (if multiple GPUs)
    - Total: n_processors × n_gpus sub-populations
    """

    PROCESSOR_BUDGETS = {
        "elm": 10,
        "mlp": 20,
        "gnn": 25,
        "transformer": 30,
        "mamba": 40,
    }

    def __init__(
        self,
        processor_types: List[str],
        gpu_ids: List[int],
        n_vars: int,
        population_per_island: int = 8,
        intra_migration_interval: int = 2,
        inter_migration_interval: int = 4,
        seed: int = 42,
        verbose: bool = True,
        use_spectral_constraint: bool = True,
        enable_pruning: bool = True,
        use_v7: bool = False,
        Y_continuous: np.ndarray = None,
    ):
        """
        Initialize hierarchical island model.

        Args:
            processor_types: List of processor types (e.g., ['elm', 'mlp', 'transformer'])
            gpu_ids: List of GPU IDs to use
            n_vars: Number of variables in causal graph
            population_per_island: Population size per island
            intra_migration_interval: Generations between intra-processor migration
            inter_migration_interval: Generations between inter-processor migration
            seed: Random seed
            verbose: Print progress
            use_spectral_constraint: Use O(d²) spectral DAG constraint
            enable_pruning: Enable dynamic edge pruning
            use_v7: Enable effect estimation with 4-objective optimization
            Y_continuous: Continuous Y for effect estimation (optional)
        """
        self.processor_types = processor_types
        self.gpu_ids = gpu_ids
        self.n_vars = n_vars
        self.pop_per_island = population_per_island
        self.intra_interval = intra_migration_interval
        self.inter_interval = inter_migration_interval
        self.seed = seed
        self.verbose = verbose
        self.use_spectral_constraint = use_spectral_constraint
        self.enable_pruning = enable_pruning
        self.use_v7 = use_v7
        self.Y_continuous = Y_continuous

        # Create islands: processor_type → [GPU shard 0, GPU shard 1, ...]
        self.islands: Dict[str, List[Island]] = {}
        island_seed = seed

        for proc in processor_types:
            self.islands[proc] = []
            for gpu_id in gpu_ids:
                island = Island(
                    config=IslandConfig(
                        processor_type=proc,
                        gpu_id=gpu_id,
                        population_size=population_per_island,
                        golem_iterations=self.PROCESSOR_BUDGETS.get(proc, 20),
                    ),
                    n_vars=n_vars,
                    seed=island_seed,
                )
                self.islands[proc].append(island)
                island_seed += 1

        # Global structure pool for inter-processor migration
        self.structure_pool: List[dict] = []
        self.structure_pool_max_size = 20

        # Statistics
        self.generation = 0
        self.stats = {
            "intra_migrations": 0,
            "inter_migrations": 0,
            "best_accuracy_per_gen": [],
        }

        if verbose:
            total_pop = self.get_total_population()
            backend = jax.default_backend()
            print("Initialized HierarchicalIslandModel:")
            print(f"  Processors: {processor_types}")
            print(f"  Backend: {backend}")
            print(f"  Device IDs: {gpu_ids}")
            print(f"  Islands: {len(processor_types) * len(gpu_ids)}")
            print(f"  Total population: {total_pop}")

    def get_total_population(self) -> int:
        """Total individuals across all islands."""
        return len(self.processor_types) * len(self.gpu_ids) * self.pop_per_island

    def should_migrate_intra(self, generation: int) -> bool:
        """Check if intra-processor migration should occur."""
        return generation > 0 and generation % self.intra_interval == 0

    def should_migrate_inter(self, generation: int) -> bool:
        """Check if inter-processor migration should occur."""
        return generation > 0 and generation % self.inter_interval == 0

    def intra_processor_migration(self):
        """
        Migrate between GPU shards of same processor type.

        Full genome transfer (same architecture).
        """
        for proc_type, shards in self.islands.items():
            if len(shards) < 2:
                continue

            # Collect elites from all shards
            all_elites = []
            for shard in shards:
                elites = shard.get_elites(n=2)
                all_elites.extend(elites)

            # Distribute elites round-robin to shards
            for i, shard in enumerate(shards):
                incoming = all_elites[i :: len(shards)]
                shard.accept_migrants(incoming, replace_worst=True)

        self.stats["intra_migrations"] += 1

    def inter_processor_migration(self):
        """
        Migrate structures between processor types.

        Only transfers:
        - Adjacency matrix structure (topology)
        - GOLEM hyperparameters (λ₁, λ₂, lr)

        Does NOT transfer:
        - Processor-specific weights (incompatible)
        """
        # Step 1: Collect best structures from each processor type
        for proc_type, shards in self.islands.items():
            best_individual = None
            best_fitness = -float("inf")

            for shard in shards:
                elite = shard.get_best()
                if elite and elite.fitness > best_fitness:
                    best_fitness = elite.fitness
                    best_individual = elite

            if best_individual and best_fitness > 0.5:  # Only share good structures
                structure_info = {
                    "A_topology": best_individual.A_topology.copy(),
                    "lambda_1": best_individual.lambda_1,
                    "lambda_2": best_individual.lambda_2,
                    "lr": best_individual.lr,
                    "source_processor": proc_type,
                    "fitness": best_fitness,
                }
                self._add_to_structure_pool(structure_info)

        # Step 2: Inject structures into each processor type
        for target_proc, shards in self.islands.items():
            # Get structure from different processor
            foreign_structure = self._get_foreign_structure(exclude=target_proc)

            if foreign_structure:
                # Create new individual with foreign structure
                for shard in shards:
                    new_individual = shard.create_from_structure(
                        A_topology=foreign_structure["A_topology"],
                        lambda_1=foreign_structure["lambda_1"],
                        lambda_2=foreign_structure["lambda_2"],
                        lr=foreign_structure["lr"],
                    )
                    shard.inject_migrant(new_individual)

        self.stats["inter_migrations"] += 1

    def _add_to_structure_pool(self, structure: dict):
        """Add structure to global pool (keep best)."""
        self.structure_pool.append(structure)

        # Sort by fitness, keep top
        self.structure_pool.sort(key=lambda x: x["fitness"], reverse=True)
        self.structure_pool = self.structure_pool[: self.structure_pool_max_size]

    def _get_foreign_structure(self, exclude: str) -> Optional[dict]:
        """Get best structure from different processor type."""
        candidates = [s for s in self.structure_pool if s["source_processor"] != exclude]
        if not candidates:
            return None
        # Return best (already sorted)
        return candidates[0]

    def run_generation(
        self, X_train: np.ndarray, Y_train: np.ndarray, parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run one generation across all islands.

        Args:
            X_train: Training features
            Y_train: Training labels
            parallel: Run islands in parallel (one per GPU)

        Returns:
            Dictionary with generation results
        """
        results = {
            "generation": self.generation,
            "processor_results": {},
            "best_accuracy": 0.0,
            "best_processor": None,
        }

        if parallel and len(self.gpu_ids) > 1:
            # Parallel evaluation across GPUs
            with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
                futures = {}

                for proc_type, shards in self.islands.items():
                    for shard in shards:
                        future = executor.submit(
                            shard.evaluate,
                            X_train,
                            Y_train,
                            verbose=self.verbose,
                            use_spectral_constraint=self.use_spectral_constraint,
                            enable_pruning=self.enable_pruning,
                            use_v7=self.use_v7,
                            Y_continuous=self.Y_continuous,
                        )
                        futures[future] = (proc_type, shard.config.gpu_id)

                for future in as_completed(futures):
                    proc_type, gpu_id = futures[future]
                    try:
                        shard_result = future.result()
                        key = f"{proc_type}_gpu{gpu_id}"
                        results["processor_results"][key] = shard_result

                        if shard_result["best_accuracy"] > results["best_accuracy"]:
                            results["best_accuracy"] = shard_result["best_accuracy"]
                            results["best_processor"] = proc_type
                    except Exception as e:
                        if self.verbose:
                            backend_name = jax.default_backend().upper()
                            print(f"  {proc_type} {backend_name} {gpu_id} failed: {e}")
        else:
            # Sequential evaluation
            for proc_type, shards in self.islands.items():
                for shard in shards:
                    shard_result = shard.evaluate(
                        X_train,
                        Y_train,
                        verbose=self.verbose,
                        use_spectral_constraint=self.use_spectral_constraint,
                        enable_pruning=self.enable_pruning,
                        use_v7=self.use_v7,
                        Y_continuous=self.Y_continuous,
                    )
                    key = f"{proc_type}_gpu{shard.config.gpu_id}"
                    results["processor_results"][key] = shard_result

                    if shard_result["best_accuracy"] > results["best_accuracy"]:
                        results["best_accuracy"] = shard_result["best_accuracy"]
                        results["best_processor"] = proc_type

        # Selection and reproduction
        for proc_type, shards in self.islands.items():
            for shard in shards:
                shard.select_and_reproduce()

        # Migration
        if self.should_migrate_intra(self.generation):
            self.intra_processor_migration()
            if self.verbose:
                print(f"  Gen {self.generation}: Intra-processor migration")

        if self.should_migrate_inter(self.generation):
            self.inter_processor_migration()
            if self.verbose:
                print(f"  Gen {self.generation}: Inter-processor migration")

        # Update stats
        self.stats["best_accuracy_per_gen"].append(results["best_accuracy"])
        self.generation += 1

        if self.verbose:
            print(
                f"Gen {self.generation - 1}: Best accuracy={results['best_accuracy']:.4f} "
                f"({results['best_processor']})"
            )

        return results

    def get_global_pareto_front(self) -> List[Individual]:
        """Merge Pareto fronts from all islands."""
        all_solutions = []

        for proc_type, shards in self.islands.items():
            for shard in shards:
                solutions = shard.get_pareto_front()
                for sol in solutions:
                    sol.processor_type = proc_type  # Tag with processor
                    all_solutions.append(sol)

        # Non-dominated sorting across all
        return self._extract_global_pareto(all_solutions)

    def _extract_global_pareto(self, solutions: List[Individual]) -> List[Individual]:
        """Extract non-dominated solutions across all processors."""
        pareto = []
        for sol in solutions:
            dominated = False
            for other in solutions:
                if other is not sol and self._dominates(other, sol):
                    dominated = True
                    break
            if not dominated:
                pareto.append(sol)
        return pareto

    @staticmethod
    def _dominates(a: Individual, b: Individual) -> bool:
        """Check if solution a dominates solution b."""
        better_in_all = a.accuracy >= b.accuracy and a.sparsity >= b.sparsity and a.h_A <= b.h_A
        strictly_better_in_one = a.accuracy > b.accuracy or a.sparsity > b.sparsity or a.h_A < b.h_A
        return better_in_all and strictly_better_in_one

    def get_best_solution(self) -> Optional[Individual]:
        """Get overall best solution across all islands."""
        best = None
        best_fitness = -float("inf")

        for proc_type, shards in self.islands.items():
            for shard in shards:
                elite = shard.get_best()
                if elite and elite.fitness > best_fitness:
                    best_fitness = elite.fitness
                    best = elite

        return best

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            **self.stats,
            "generation": self.generation,
            "structure_pool_size": len(self.structure_pool),
            "best_accuracy": max(self.stats["best_accuracy_per_gen"])
            if self.stats["best_accuracy_per_gen"]
            else 0.0,
        }


# ============================================================================
# Convenience function for running island-model experiments
# ============================================================================


def run_island_model_experiment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    processor_types: List[str] = ["elm", "mlp", "transformer"],
    gpu_ids: List[int] = [0, 1],
    n_generations: int = 15,
    population_per_island: int = 8,
    seed: int = 42,
    verbose: bool = True,
    use_spectral_constraint: bool = True,
    enable_pruning: bool = True,
    use_v7: bool = False,
    Y_continuous: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Run complete island-model experiment.

    Args:
        X_train, Y_train: Training data (Y_train is binary for classification)
        X_test, Y_test: Test data
        processor_types: Processor types to use
        gpu_ids: GPU IDs to use
        n_generations: Number of generations
        population_per_island: Population per island
        seed: Random seed
        verbose: Print progress
        use_spectral_constraint: Use O(d²) spectral DAG constraint
        enable_pruning: Enable dynamic edge pruning
        use_v7: Enable effect estimation with 4-objective optimization
        Y_continuous: Continuous Y for effect estimation (optional)

    Returns:
        Results dictionary with Pareto front, best solution, and stats
    """
    n_features = X_train.shape[1]
    n_vars = n_features  # Number of X features (GOLEM adds +1 for Y internally)

    model = HierarchicalIslandModel(
        processor_types=processor_types,
        gpu_ids=gpu_ids,
        n_vars=n_vars,
        population_per_island=population_per_island,
        seed=seed,
        verbose=verbose,
        use_spectral_constraint=use_spectral_constraint,
        enable_pruning=enable_pruning,
        use_v7=use_v7,
        Y_continuous=Y_continuous,
    )

    # Run generations
    for gen in range(n_generations):
        results = model.run_generation(X_train, Y_train)

    # Get final results
    pareto_front = model.get_global_pareto_front()
    best_solution = model.get_best_solution()
    stats = model.get_stats()

    # Evaluate best solution on test set
    if best_solution:
        from sklearn.linear_model import LogisticRegression

        A_np = best_solution.A_topology
        Y_idx = n_features
        mb_indices = np.where(np.abs(A_np[:, Y_idx]) > 0.01)[0]
        mb_indices = mb_indices[mb_indices < n_features]

        if len(mb_indices) > 0:
            X_mb_train = X_train[:, mb_indices]
            X_mb_test = X_test[:, mb_indices]
        else:
            X_mb_train = X_train
            X_mb_test = X_test

        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_mb_train, Y_train)
        test_accuracy = clf.score(X_mb_test, Y_test)
    else:
        test_accuracy = 0.0

    return {
        "pareto_front": pareto_front,
        "best_solution": best_solution,
        "test_accuracy": test_accuracy,
        "stats": stats,
        "n_pareto_solutions": len(pareto_front),
    }
