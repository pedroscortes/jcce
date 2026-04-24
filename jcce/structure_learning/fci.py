"""
FCI (Fast Causal Inference) Algorithm for Causal Discovery.

Constraint-based algorithm that handles latent (hidden) confounders.
Unlike PC which assumes causal sufficiency (no hidden variables),
FCI outputs a PAG (Partial Ancestral Graph) that represents an
equivalence class of MAGs (Maximal Ancestral Graphs).

References:
    - Spirtes, P., Meek, C., & Richardson, T. (1999).
      "An algorithm for causal inference in the presence of latent
      variables and selection bias."
    - Zhang, J. (2008). "On the completeness of orientation rules
      for causal discovery in the presence of latent confounders
      and selection bias." Artificial Intelligence, 172(16-17), 1873-1896.
    - Colombo, D., Maathuis, M. H., Kalisch, M., & Richardson, T. S. (2012).
      "Learning high-dimensional directed acyclic graphs with latent and
      selection variables." The Annals of Statistics, 40(1), 294-321.

PAG Encoding (adjacency matrix pag[i,j] = mark at j on the i-j edge):
    0 = no edge
    1 = circle (o)   — unknown: could be tail or arrowhead
    2 = arrowhead (>) — definite arrowhead
    3 = tail (-)      — definite tail

Common edge patterns:
    i -> j  (definite cause):    pag[i,j]=2, pag[j,i]=3
    i <-> j (latent confounder): pag[i,j]=2, pag[j,i]=2
    i o-> j (possible cause):    pag[i,j]=2, pag[j,i]=1
    i o-o j (unknown):           pag[i,j]=1, pag[j,i]=1

Algorithm Steps:
    1. Skeleton learning (same as PC — CI tests with increasing cond. set size)
    2. Possible D-Sep: remove additional edges using d-separation reasoning
    3. Re-orient v-structures on the thinned skeleton
    4. Apply FCI orientation rules R1-R4, R8-R10 (Zhang 2008)
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Optional, Tuple, Set, Dict, List
from itertools import combinations
import numpy as np

from jcce.structure_learning.pc import (
    partial_correlation_jax,
    fisher_z_test,
    conditional_independence_test,
)


# PAG edge marks
NONE = 0
CIRCLE = 1
ARROW = 2
TAIL = 3

# Human-readable labels
_MARK_NAMES = {NONE: ' ', CIRCLE: 'o', ARROW: '>', TAIL: '-'}


# ============================================================================
# Phase 1: Skeleton Learning (same as PC)
# ============================================================================

def _learn_skeleton(
    data: jnp.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Learn skeleton via conditional independence tests (identical to PC Phase 1).

    Returns:
        skeleton: (n_vars, n_vars) numpy undirected adjacency {0, 1}
        sepsets: dict mapping (i, j) -> conditioning set that separated them
    """
    n_samples, n_vars = data.shape

    if verbose:
        print(f"  Phase 1: Skeleton learning ({n_vars} vars, {n_samples} samples)")

    skeleton = np.ones((n_vars, n_vars), dtype=np.float64)
    np.fill_diagonal(skeleton, 0.0)

    sepsets: Dict[Tuple[int, int], jnp.ndarray] = {}

    for cond_size in range(max_cond_size + 1):
        if verbose:
            print(f"    CI tests with |S| = {cond_size}")

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] == 0:
                    continue

                neighbors_i = [
                    k for k in range(n_vars)
                    if skeleton[i, k] == 1 and k != j
                ]

                if len(neighbors_i) < cond_size:
                    continue

                for cond_set_tuple in combinations(neighbors_i, cond_size):
                    cond_set = jnp.array(list(cond_set_tuple), dtype=jnp.int32)

                    if conditional_independence_test(data, i, j, cond_set, alpha):
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        sepsets[(i, j)] = cond_set
                        sepsets[(j, i)] = cond_set
                        break

    if verbose:
        n_edges = int(skeleton.sum()) // 2
        print(f"    Skeleton: {n_edges} undirected edges")

    return skeleton, sepsets


# ============================================================================
# Phase 2: Possible D-Sep
# ============================================================================

def _get_neighbors(skeleton: np.ndarray, node: int) -> List[int]:
    """Get neighbors of node in the skeleton."""
    n = skeleton.shape[0]
    return [j for j in range(n) if skeleton[node, j] != 0]


def _possible_dsep(
    skeleton: np.ndarray,
    initial_pdag: np.ndarray,
    x: int,
    y: int,
) -> Set[int]:
    """
    Compute Possible-D-Sep(x, y).

    A node v is in PDS(x, y) if there exists a path <x, u1, ..., v> in the
    skeleton such that for every consecutive triple <a, b, c> on the path,
    b is a collider on the path OR b is adjacent to x.

    This is a conservative approximation using BFS: we traverse from x,
    continuing through node b if b is adjacent to x or if b has an
    arrowhead pointing at it from the initial v-structure orientation
    (i.e., b could be a collider).

    Args:
        skeleton: (n, n) undirected adjacency
        initial_pdag: (n, n) PDAG from initial v-structure orientation
                      (used to detect potential colliders)
        x: source node
        y: other endpoint (excluded from result)

    Returns:
        Set of node indices in PDS(x, y), excluding x and y.
    """
    n = skeleton.shape[0]
    pds: Set[int] = set()
    visited: Set[int] = {x}

    # BFS: (current_node, previous_node)
    queue: List[Tuple[int, int]] = []

    for nb in _get_neighbors(skeleton, x):
        if nb != y:
            queue.append((nb, x))
            visited.add(nb)
            pds.add(nb)

    while queue:
        current, prev = queue.pop(0)

        # Continue through `current` if:
        # (a) current is adjacent to x (safe to condition on), OR
        # (b) current could be a collider (has arrowheads pointing in)
        adj_to_x = skeleton[x, current] != 0
        could_be_collider = False
        if initial_pdag is not None:
            # Check if any neighbor points an arrow into current
            for k in range(n):
                if k != current and initial_pdag[k, current] == 1:
                    could_be_collider = True
                    break

        if adj_to_x or could_be_collider:
            for nb in _get_neighbors(skeleton, current):
                if nb not in visited and nb != y:
                    visited.add(nb)
                    pds.add(nb)
                    queue.append((nb, current))

    pds.discard(x)
    pds.discard(y)
    return pds


def _possible_dsep_phase(
    data: jnp.ndarray,
    skeleton: np.ndarray,
    sepsets: dict,
    initial_pdag: np.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 4,
    max_pds_size: int = -1,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    FCI Phase 2: Remove additional edges using possible d-separating sets.

    For each adjacent pair (i, j), compute PDS(i, j) ∪ PDS(j, i) and test
    CI conditioning on subsets. If CI found, remove the edge and record
    the separating set.

    Args:
        data: (n_samples, n_vars) data matrix
        skeleton: (n_vars, n_vars) undirected adjacency from Phase 1
        sepsets: separating sets from Phase 1
        initial_pdag: PDAG with v-structures (for collider detection in PDS)
        alpha: CI test significance level
        max_cond_size: max conditioning set size to test
        max_pds_size: max PDS size to consider (-1 = no limit)
        verbose: print progress

    Returns:
        Updated (skeleton, sepsets)
    """
    n_vars = skeleton.shape[0]
    skel = skeleton.copy()

    if verbose:
        print("  Phase 2: Possible D-Sep")

    edges_removed = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skel[i, j] == 0:
                continue

            pds_i = _possible_dsep(skel, initial_pdag, i, j)
            pds_j = _possible_dsep(skel, initial_pdag, j, i)
            pds = sorted(pds_i | pds_j)

            if max_pds_size > 0 and len(pds) > max_pds_size:
                pds = pds[:max_pds_size]

            if not pds:
                continue

            found = False
            for size in range(min(len(pds), max_cond_size) + 1):
                if found:
                    break
                for cond_tuple in combinations(pds, size):
                    cond_set = jnp.array(list(cond_tuple), dtype=jnp.int32)

                    if conditional_independence_test(data, i, j, cond_set, alpha):
                        skel[i, j] = 0
                        skel[j, i] = 0
                        sepsets[(i, j)] = cond_set
                        sepsets[(j, i)] = cond_set
                        edges_removed += 1
                        found = True
                        break

    if verbose:
        n_edges = int(skel.sum()) // 2
        print(f"    Removed {edges_removed} additional edges")
        print(f"    Thinned skeleton: {n_edges} undirected edges")

    return skel, sepsets


# ============================================================================
# Phase 3: Orient V-Structures (PAG encoding)
# ============================================================================

def _orient_v_structures_pag(
    skeleton: np.ndarray,
    sepsets: dict,
    verbose: bool = False,
) -> np.ndarray:
    """
    Orient v-structures and initialize PAG with circle marks.

    Same logic as PC v-structure detection, but uses PAG encoding:
    - All remaining edges start as o-o (circle at both ends)
    - V-structures get arrowheads: i *-> k <-* j

    Args:
        skeleton: (n_vars, n_vars) undirected adjacency
        sepsets: separating sets

    Returns:
        pag: (n_vars, n_vars) PAG with encoding {0, 1, 2, 3}
    """
    n_vars = skeleton.shape[0]

    # Initialize: all edges as o-o (circle at both ends)
    pag = np.zeros((n_vars, n_vars), dtype=np.int32)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton[i, j] != 0:
                pag[i, j] = CIRCLE
                pag[j, i] = CIRCLE

    if verbose:
        print("  Phase 3: Orient v-structures (PAG)")

    n_oriented = 0

    for k in range(n_vars):
        # Find all pairs (i, j) that are both adjacent to k but not to each other
        adj_k = [v for v in range(n_vars) if v != k and _has_edge(pag, v, k)]

        for idx_a in range(len(adj_k)):
            for idx_b in range(idx_a + 1, len(adj_k)):
                i, j = adj_k[idx_a], adj_k[idx_b]

                # i and j must NOT be adjacent
                if _has_edge(pag, i, j):
                    continue

                # k must NOT be in sepset(i, j)
                sepset_ij = sepsets.get((i, j), jnp.array([], dtype=jnp.int32))
                k_in_sepset = int(k) in [int(s) for s in sepset_ij]

                if not k_in_sepset:
                    # Orient as i *-> k <-* j
                    pag[i, k] = ARROW
                    pag[j, k] = ARROW
                    n_oriented += 1

    if verbose:
        print(f"    Oriented {n_oriented} v-structures")

    return pag


# ============================================================================
# Phase 4: FCI Orientation Rules (Zhang 2008)
# ============================================================================

def _has_edge(pag: np.ndarray, i: int, j: int) -> bool:
    """Check if any edge exists between i and j."""
    return pag[i, j] != NONE or pag[j, i] != NONE


def _rule_r1(pag: np.ndarray, n: int) -> bool:
    """
    R1: If a *-> b o-* c, and a,c not adjacent, orient b -> c.

    Intuition: a's arrow into b means b is an ancestor of a's non-descendant,
    so b must also be an ancestor of c (via the b-c edge), meaning b -> c.

    Condition: pag[a,b]=ARROW, pag[c,b]=CIRCLE, pag[b,c]!=NONE,
               no edge between a and c.
    Action:    pag[b,c]=ARROW, pag[c,b]=TAIL
    """
    changed = False
    for a in range(n):
        for b in range(n):
            if a == b or pag[a, b] != ARROW:
                continue
            for c in range(n):
                if c == a or c == b:
                    continue
                if pag[c, b] != CIRCLE:
                    continue
                if pag[b, c] == NONE:
                    continue
                if _has_edge(pag, a, c):
                    continue
                pag[b, c] = ARROW
                pag[c, b] = TAIL
                changed = True
    return changed


def _rule_r2(pag: np.ndarray, n: int) -> bool:
    """
    R2: If a -> b *-> c (or a *-> b -> c), and a *-o c, orient a *-> c.

    Intuition: there is a directed path a -> ... -> c, so a must be an
    ancestor of c — the circle at c becomes an arrowhead.

    Sub-case 1: a -> b *-> c with a *-o c
        pag[a,b]=ARROW, pag[b,a]=TAIL, pag[b,c]=ARROW, pag[a,c]=CIRCLE
    Sub-case 2: a *-> b -> c with a *-o c
        pag[a,b]=ARROW, pag[b,c]=ARROW, pag[c,b]=TAIL, pag[a,c]=CIRCLE

    Action: pag[a,c] = ARROW
    """
    changed = False
    for a in range(n):
        for c in range(n):
            if a == c or pag[a, c] != CIRCLE:
                continue
            if not _has_edge(pag, a, c):
                continue
            for b in range(n):
                if b == a or b == c:
                    continue
                # Sub-case 1: a -> b *-> c
                case1 = (
                    pag[a, b] == ARROW and pag[b, a] == TAIL and
                    pag[b, c] == ARROW
                )
                # Sub-case 2: a *-> b -> c
                case2 = (
                    pag[a, b] == ARROW and
                    pag[b, c] == ARROW and pag[c, b] == TAIL
                )
                if case1 or case2:
                    pag[a, c] = ARROW
                    changed = True
                    break
    return changed


def _rule_r3(pag: np.ndarray, n: int) -> bool:
    """
    R3: If a *-> b <-* c, a *-o d, c *-o d, a and c not adjacent,
        d *-o b, orient d *-> b.

    Intuition: d is connected to two non-adjacent parents of b (a and c),
    forming a potential collider structure at b through d.

    Condition:
        pag[a,b]=ARROW, pag[c,b]=ARROW,
        not _has_edge(a,c),
        _has_edge(a,d), _has_edge(c,d),
        pag[d,b]=CIRCLE (circle at b from d's side)
    Action: pag[d,b] = ARROW
    """
    changed = False
    for d in range(n):
        for b in range(n):
            if d == b or pag[d, b] != CIRCLE:
                continue
            if not _has_edge(pag, d, b):
                continue

            # Find pairs (a, c) that both have arrows into b
            arrows_into_b = [
                v for v in range(n) if v != b and v != d and pag[v, b] == ARROW
            ]

            for idx_a in range(len(arrows_into_b)):
                for idx_c in range(idx_a + 1, len(arrows_into_b)):
                    a, c = arrows_into_b[idx_a], arrows_into_b[idx_c]

                    if _has_edge(pag, a, c):
                        continue
                    if not _has_edge(pag, a, d):
                        continue
                    if not _has_edge(pag, c, d):
                        continue
                    # R3 requires d *-o a and d *-o c (circle at a/c from d's side)
                    if pag[d, a] != CIRCLE:
                        continue
                    if pag[d, c] != CIRCLE:
                        continue

                    pag[d, b] = ARROW
                    changed = True
    return changed


def _rule_r4(pag: np.ndarray, sepsets: dict, n: int) -> bool:
    """
    R4: Discriminating path rule.

    A discriminating path for b between a and c is:
        <a, ..., v, b, c> where:
        - a is not adjacent to c
        - every node between a and c (except b) is a parent of c
        - b is adjacent to c

    If such a path exists:
        - If b in sepset(a, c): orient b -> c (b is a non-collider)
        - If b not in sepset(a, c): orient <v, b, c> as v <-> b <-> c
          (b is a collider on the discriminating path)

    We search for discriminating paths of length >= 3 via BFS.
    """
    changed = False

    for c in range(n):
        for b in range(n):
            if b == c or not _has_edge(pag, b, c):
                continue
            if pag[b, c] != CIRCLE:
                continue

            # b must have an arrow into it from c's side: pag[c,b] can be anything
            # Actually: for discriminating path, b o-* c with circle at c
            # means pag[b,c] = CIRCLE. And we need the edge b-c.

            # Search for discriminating paths ending at <..., v, b, c>
            # where every intermediate node is a parent of c (arrow into c, tail out)
            # i.e., pag[node, c] = ARROW and pag[c, node] = TAIL

            # BFS backward from b to find path <a, ..., v, b>
            # where every intermediate node (between a and b) is a parent of c

            # Parents of c (definite: -> c)
            parents_c = set()
            for v in range(n):
                if v != c and pag[v, c] == ARROW and pag[c, v] == TAIL:
                    parents_c.add(v)

            # BFS: find nodes reachable from b through parents of c
            # path_prev[v] = predecessor on path from some `a` to v
            visited = {b, c}
            queue: List[Tuple[int, int]] = []  # (current, prev)

            # Start: neighbors of b that are parents of c
            for v in _get_adj(pag, b, n):
                if v != c and v in parents_c and v not in visited:
                    # v must have arrow at b: pag[v,b] == ARROW
                    if pag[v, b] == ARROW:
                        queue.append((v, b))
                        visited.add(v)

            found_path = False
            while queue and not found_path:
                current, prev = queue.pop(0)

                # Check: can `current` be the start `a` of a discriminating path?
                # `a` must NOT be adjacent to c and must NOT be a parent of c
                # (a is the start, not an intermediate)
                if not _has_edge(pag, current, c):
                    # Found discriminating path: <current, ..., prev, b, c>
                    a = current
                    sepset_ac = sepsets.get(
                        (a, c), sepsets.get((c, a), jnp.array([], dtype=jnp.int32))
                    )
                    b_in_sepset = int(b) in [int(s) for s in sepset_ac]

                    if b_in_sepset:
                        # b is a non-collider: orient b -> c
                        pag[b, c] = ARROW
                        pag[c, b] = TAIL
                    else:
                        # b is a collider: orient v <-> b <-> c
                        pag[b, c] = ARROW
                        pag[c, b] = ARROW

                    changed = True
                    found_path = True
                    break

                # Continue search: current must be a parent of c
                if current not in parents_c:
                    continue

                for nb in _get_adj(pag, current, n):
                    if nb not in visited and nb != c:
                        if nb in parents_c or not _has_edge(pag, nb, c):
                            if pag[nb, current] == ARROW or pag[current, nb] == ARROW:
                                visited.add(nb)
                                queue.append((nb, current))

    return changed


def _get_adj(pag: np.ndarray, node: int, n: int) -> List[int]:
    """Get all adjacent nodes in the PAG."""
    return [j for j in range(n) if j != node and _has_edge(pag, node, j)]


def _rule_r8(pag: np.ndarray, n: int) -> bool:
    """
    R8: If a -> b -> c (or a -o b -> c), and a o-> c, orient a -> c.

    Intuition: there's a directed path a -> b -> c, and a o-> c means
    the circle at a could be a tail. Since a -> b -> c implies a is
    an ancestor of c, the a-c edge must be a -> c (not a <-> c).

    Condition:
        pag[b,c]=ARROW, pag[c,b]=TAIL (b -> c)
        pag[a,b]=ARROW (or just edge), pag[b,a]=TAIL or pag[b,a]=CIRCLE
            i.e., a -> b (tail+arrow) or a -o b (tail at a? no...)
        Actually: a -> b means pag[a,b]=ARROW, pag[b,a]=TAIL
                  a -o b means pag[a,b]=CIRCLE, pag[b,a]=TAIL
        Wait. "a -o b" = tail at a, circle at b.
        pag[b,a] = TAIL (mark at a = tail), pag[a,b] = CIRCLE (mark at b = circle)?
        No. pag[a,b] = mark at b on a-b edge.
        "a -o b" = - at a, o at b. Mark at a = -, mark at b = o.
        pag[b,a] = TAIL (mark at a), pag[a,b] = CIRCLE (mark at b). Hmm...

        Actually: a -> b means: tail at a, arrow at b.
        pag[a,b] = ARROW (mark at b = >), pag[b,a] = TAIL (mark at a = -)

        a -o b means: tail at a, circle at b.
        pag[a,b] = CIRCLE (mark at b = o), pag[b,a] = TAIL (mark at a = -)

    R8 condition: (a -> b OR a -o b) AND b -> c AND a o-> c
        a -> b: pag[a,b]=ARROW, pag[b,a]=TAIL
        a -o b: pag[a,b]=CIRCLE, pag[b,a]=TAIL
        Combined: pag[b,a]=TAIL, pag[a,b] in {ARROW, CIRCLE}

        b -> c: pag[b,c]=ARROW, pag[c,b]=TAIL

        a o-> c: pag[a,c]=ARROW, pag[c,a]=CIRCLE

    Action: orient a -> c: pag[c,a] = TAIL (circle at a becomes tail)
    """
    changed = False
    for a in range(n):
        for c in range(n):
            if a == c:
                continue
            # a o-> c
            if pag[a, c] != ARROW or pag[c, a] != CIRCLE:
                continue

            for b in range(n):
                if b == a or b == c:
                    continue
                # b -> c
                if pag[b, c] != ARROW or pag[c, b] != TAIL:
                    continue
                # a -> b or a -o b: pag[b,a]=TAIL
                if pag[b, a] != TAIL:
                    continue
                if pag[a, b] not in (ARROW, CIRCLE):
                    continue

                pag[c, a] = TAIL
                changed = True
                break
    return changed


def _find_uncovered_pd_path(
    pag: np.ndarray,
    a: int,
    c: int,
    n: int,
    max_length: int = 20,
) -> Optional[List[int]]:
    """
    Find an uncovered potentially directed path from a to c.

    Potentially directed: for each edge on the path, the mark at the
    node closer to a is not an arrowhead (i.e., not pointing back).

    Uncovered: for every triple <x, y, z> on the path, x and z are
    not adjacent.

    Uses BFS. Returns the path as list of nodes, or None if not found.
    """
    # BFS: (current_node, path_so_far)
    queue: List[Tuple[int, List[int]]] = [(a, [a])]
    visited: Set[int] = {a}

    while queue:
        current, path = queue.pop(0)

        if len(path) > max_length:
            continue

        if current == c and len(path) >= 3:
            return path

        for nb in range(n):
            if nb == current or nb in visited:
                continue
            if not _has_edge(pag, current, nb):
                continue

            # Potentially directed: mark at current (closer to a) is not arrowhead
            # pag[nb, current] = mark at current on the nb-current edge
            if pag[nb, current] == ARROW:
                continue

            # Uncovered: if path has >= 2 nodes, check that prev and nb
            # are not adjacent
            if len(path) >= 2:
                prev = path[-2]
                if _has_edge(pag, prev, nb):
                    continue

            visited.add(nb)
            queue.append((nb, path + [nb]))

    return None


def _rule_r9(pag: np.ndarray, n: int) -> bool:
    """
    R9: If a o-> c, and there exists an uncovered potentially directed
        path <a, b, ..., c> where b and c are not adjacent,
        orient a -> c.

    Action: pag[c,a] = TAIL
    """
    changed = False
    for a in range(n):
        for c in range(n):
            if a == c:
                continue
            if pag[a, c] != ARROW or pag[c, a] != CIRCLE:
                continue

            path = _find_uncovered_pd_path(pag, a, c, n)
            if path is not None and len(path) >= 3:
                b = path[1]
                if not _has_edge(pag, b, c):
                    pag[c, a] = TAIL
                    changed = True
    return changed


def _rule_r10(pag: np.ndarray, n: int) -> bool:
    """
    R10: If a o-> c, b -> c <- d, there exist uncovered potentially
         directed paths <a, ..., mu, b> and <a, ..., omega, d>,
         mu and omega are distinct and not adjacent,
         orient a -> c.

    Action: pag[c,a] = TAIL
    """
    changed = False
    for a in range(n):
        for c in range(n):
            if a == c:
                continue
            if pag[a, c] != ARROW or pag[c, a] != CIRCLE:
                continue

            # Find b, d with b -> c and d -> c
            parents_of_c = [
                v for v in range(n)
                if v != c and v != a and pag[v, c] == ARROW and pag[c, v] == TAIL
            ]

            oriented = False
            for idx_b in range(len(parents_of_c)):
                if oriented:
                    break
                for idx_d in range(idx_b + 1, len(parents_of_c)):
                    b = parents_of_c[idx_b]
                    d = parents_of_c[idx_d]

                    if _has_edge(pag, b, d):
                        continue

                    path_ab = _find_uncovered_pd_path(pag, a, b, n)
                    if path_ab is None or len(path_ab) < 3:
                        continue

                    path_ad = _find_uncovered_pd_path(pag, a, d, n)
                    if path_ad is None or len(path_ad) < 3:
                        continue

                    mu = path_ab[-2]     # node before b on path a->b
                    omega = path_ad[-2]  # node before d on path a->d

                    if mu != omega and not _has_edge(pag, mu, omega):
                        pag[c, a] = TAIL
                        changed = True
                        oriented = True
                        break
    return changed


def _apply_fci_rules(
    pag: np.ndarray,
    sepsets: dict,
    max_iter: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply FCI orientation rules R1-R4, R8-R10 iteratively until convergence.
    """
    n = pag.shape[0]

    if verbose:
        print("  Phase 4: FCI orientation rules (R1-R4, R8-R10)")

    for iteration in range(max_iter):
        changed = False

        changed |= _rule_r1(pag, n)
        changed |= _rule_r2(pag, n)
        changed |= _rule_r3(pag, n)
        changed |= _rule_r4(pag, sepsets, n)
        changed |= _rule_r8(pag, n)
        changed |= _rule_r9(pag, n)
        changed |= _rule_r10(pag, n)

        if not changed:
            break

    if verbose:
        print(f"    Converged in {iteration + 1} iterations")

    return pag


# ============================================================================
# Initial V-Structure PDAG (for Possible D-Sep computation)
# ============================================================================

def _initial_v_structure_pdag(
    skeleton: np.ndarray,
    sepsets: dict,
) -> np.ndarray:
    """
    Quick v-structure orientation on the initial skeleton (before possible
    d-sep thinning). Used only to inform the PDS computation about potential
    colliders. Returns a simple PDAG (1 = directed, -1 = undirected).
    """
    n = skeleton.shape[0]
    pdag = np.where(skeleton != 0, -1, 0).astype(np.float64)

    for k in range(n):
        adj_k = [v for v in range(n) if skeleton[v, k] != 0 and v != k]
        for idx_a in range(len(adj_k)):
            for idx_b in range(idx_a + 1, len(adj_k)):
                i, j = adj_k[idx_a], adj_k[idx_b]
                if skeleton[i, j] != 0:
                    continue
                sepset_ij = sepsets.get((i, j), jnp.array([], dtype=jnp.int32))
                if int(k) not in [int(s) for s in sepset_ij]:
                    pdag[i, k] = 1
                    pdag[k, i] = 0
                    pdag[j, k] = 1
                    pdag[k, j] = 0

    return pdag


# ============================================================================
# Main FCI Algorithm
# ============================================================================

def learn_with_fci(
    data: jnp.ndarray,
    key: random.PRNGKey,
    alpha: float = 0.05,
    max_cond_size: int = 3,
    max_cond_size_dsep: int = 4,
    max_pds_size: int = -1,
    verbose: bool = False,
    A_prior: Optional[jnp.ndarray] = None,
    lambda_prior: float = 0.1,
) -> np.ndarray:
    """
    FCI Algorithm for causal discovery with latent confounders.

    Unlike PC, FCI does not assume causal sufficiency — it can detect
    the presence of latent common causes (L -> X, L -> Y manifests as
    X <-> Y in the PAG).

    Algorithm:
        1. Skeleton learning (CI tests, same as PC)
        2. Initial v-structure detection (for PDS computation)
        3. Possible D-Sep: remove additional edges
        4. Re-orient v-structures on thinned skeleton
        5. Apply FCI rules R1-R4, R8-R10

    Args:
        data: (n_samples, n_vars) observed data
        key: JAX random key (unused, kept for API consistency)
        alpha: Significance level for CI tests (default: 0.05)
        max_cond_size: Max conditioning set size for skeleton (default: 3)
        max_cond_size_dsep: Max conditioning set size for possible d-sep
                            phase (default: 4)
        max_pds_size: Max PDS set size to consider. -1 = no limit.
                      Set to e.g. 8 for large graphs to limit computation.
        verbose: Print progress
        A_prior: Optional prior adjacency (unused, kept for API consistency)
        lambda_prior: Weight for prior (unused, kept for API consistency)

    Returns:
        pag: (n_vars, n_vars) PAG adjacency matrix (int32).
             Encoding: 0=none, 1=circle, 2=arrowhead, 3=tail.

             Use pag_to_dag() to extract a conservative DAG, or
             extract_markov_blanket_pag() to get the MB directly.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"FCI ALGORITHM (Fast Causal Inference)")
        print(f"{'='*60}")
        n_samples, n_vars = data.shape
        print(f"Data: {n_samples} samples, {n_vars} variables")
        print(f"Alpha: {alpha}, max_cond: {max_cond_size}, "
              f"max_cond_dsep: {max_cond_size_dsep}")

    # Phase 1: Skeleton learning
    skeleton, sepsets = _learn_skeleton(
        data, alpha, max_cond_size, verbose
    )

    # Initial v-structure orientation (for PDS computation only)
    initial_pdag = _initial_v_structure_pdag(skeleton, sepsets)

    # Phase 2: Possible D-Sep — remove additional edges
    skeleton, sepsets = _possible_dsep_phase(
        data, skeleton, sepsets, initial_pdag,
        alpha, max_cond_size_dsep, max_pds_size, verbose
    )

    # Phase 3: Orient v-structures on the thinned skeleton (PAG encoding)
    pag = _orient_v_structures_pag(skeleton, sepsets, verbose)

    # Phase 4: Apply FCI orientation rules
    pag = _apply_fci_rules(pag, sepsets, max_iter=100, verbose=verbose)

    if verbose:
        stats = summarize_pag(pag)
        print(f"\n  PAG Summary:")
        print(f"    Directed (->):        {stats['n_directed']}")
        print(f"    Bidirected (<->):      {stats['n_bidirected']}")
        print(f"    Partially oriented (o->): {stats['n_partially_oriented']}")
        print(f"    Unoriented (o-o):      {stats['n_unoriented']}")
        print(f"    Total edges:           {stats['n_total_edges']}")
        print(f"{'='*60}\n")

    return pag


# ============================================================================
# PAG Utilities
# ============================================================================

def summarize_pag(pag: np.ndarray) -> dict:
    """
    Summarize edge types in a PAG.

    Returns:
        Dictionary with counts of each edge type.
    """
    n = pag.shape[0]
    n_directed = 0
    n_bidirected = 0
    n_partially_oriented = 0
    n_unoriented = 0

    for i in range(n):
        for j in range(i + 1, n):
            if pag[i, j] == NONE and pag[j, i] == NONE:
                continue
            mi, mj = pag[i, j], pag[j, i]

            if (mi == ARROW and mj == TAIL) or (mi == TAIL and mj == ARROW):
                n_directed += 1
            elif mi == ARROW and mj == ARROW:
                n_bidirected += 1
            elif (mi == ARROW and mj == CIRCLE) or (mi == CIRCLE and mj == ARROW):
                n_partially_oriented += 1
            elif mi == CIRCLE and mj == CIRCLE:
                n_unoriented += 1

    return {
        'n_directed': n_directed,
        'n_bidirected': n_bidirected,
        'n_partially_oriented': n_partially_oriented,
        'n_unoriented': n_unoriented,
        'n_total_edges': (
            n_directed + n_bidirected + n_partially_oriented + n_unoriented
        ),
    }


def pag_to_dag(pag: np.ndarray) -> jnp.ndarray:
    """
    Convert PAG to a conservative binary DAG.

    Extraction rules (A[i,j] = 1 means i -> j):
        - i -> j  (definite cause):    A[i,j] = 1
        - i <-> j (latent confounder): A[i,j] = 1, A[j,i] = 1
          (both directions, since the confounder affects both)
        - i o-> j (possible cause):    A[i,j] = 1  (conservative: include it)
        - i o-o j (unknown):           A[i,j] = 1  (lower index -> higher, like PC)

    Convention: A[i,j] = i -> j, matching all other learn_with_* functions.

    Args:
        pag: (n_vars, n_vars) PAG matrix

    Returns:
        A: (n_vars, n_vars) binary adjacency matrix (float32)
    """
    pag_np = np.array(pag)
    n = pag_np.shape[0]
    A = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            mi = pag_np[i, j]  # mark at j
            mj = pag_np[j, i]  # mark at i

            if mi == NONE and mj == NONE:
                continue

            # i -> j: arrow at j, tail at i
            if mi == ARROW and mj == TAIL:
                A[i, j] = 1.0

            # j -> i: arrow at i, tail at j
            elif mj == ARROW and mi == TAIL:
                A[j, i] = 1.0

            # i <-> j: bidirected (latent confounder)
            elif mi == ARROW and mj == ARROW:
                A[i, j] = 1.0
                A[j, i] = 1.0

            # i o-> j: possible cause (conservative: include)
            elif mi == ARROW and mj == CIRCLE:
                A[i, j] = 1.0

            # j o-> i: possible cause
            elif mj == ARROW and mi == CIRCLE:
                A[j, i] = 1.0

            # i o-o j: unknown (orient lower -> higher)
            elif mi == CIRCLE and mj == CIRCLE:
                A[i, j] = 1.0

    return jnp.array(A)


def pag_to_definite_dag(pag: np.ndarray) -> jnp.ndarray:
    """
    Convert PAG to a DAG using only definite (certain) causal edges.

    Only includes i -> j edges (ARROW at j, TAIL at i). Ignores
    bidirected, partially oriented, and unoriented edges.

    This is the most conservative extraction — only edges the algorithm
    is certain about.

    Args:
        pag: (n_vars, n_vars) PAG matrix

    Returns:
        A: (n_vars, n_vars) binary adjacency matrix (float32)
    """
    pag_np = np.array(pag)
    n = pag_np.shape[0]
    A = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pag_np[i, j] == ARROW and pag_np[j, i] == TAIL:
                A[i, j] = 1.0

    return jnp.array(A)


def extract_markov_blanket_pag(
    pag: np.ndarray,
    target: int,
    include_possible: bool = True,
) -> dict:
    """
    Extract Markov Blanket from a PAG.

    In a PAG, the MB includes:
        - Definite parents: i -> target (arrow at target, tail at i)
        - Definite children: target -> j (arrow at j, tail at target)
        - Possible parents: i o-> target (arrow at target, circle at i)
        - Possible children: target o-> j (arrow at j, circle at target)
        - Spouses: other parents/possible-parents of definite/possible children
        - Bidirected neighbors: i <-> target (latent confounder)

    Args:
        pag: (n_vars, n_vars) PAG matrix
        target: target variable index
        include_possible: If True, include possible parents/children
                          (o-> edges). If False, only definite edges.

    Returns:
        Dictionary with:
            'definite_parents': definite parents of target
            'definite_children': definite children of target
            'possible_parents': possible parents (o-> target)
            'possible_children': possible children (target o->)
            'spouses': other parents of children
            'bidirected': variables connected via <->
            'mb_all': full MB indices (sorted)
            'mb_size': size of MB
    """
    pag_np = np.array(pag)
    n = pag_np.shape[0]

    definite_parents = []
    definite_children = []
    possible_parents = []
    possible_children = []
    bidirected = []

    for v in range(n):
        if v == target:
            continue
        mark_at_target = pag_np[v, target]  # mark at target on v-target edge
        mark_at_v = pag_np[target, v]       # mark at v on target-v edge

        if mark_at_target == NONE and mark_at_v == NONE:
            continue

        # v -> target: definite parent
        if mark_at_target == ARROW and mark_at_v == TAIL:
            definite_parents.append(v)

        # target -> v: definite child
        elif mark_at_v == ARROW and mark_at_target == TAIL:
            definite_children.append(v)

        # v <-> target: bidirected (latent confounder)
        elif mark_at_target == ARROW and mark_at_v == ARROW:
            bidirected.append(v)

        # v o-> target: possible parent
        elif mark_at_target == ARROW and mark_at_v == CIRCLE:
            possible_parents.append(v)

        # target o-> v: possible child
        elif mark_at_v == ARROW and mark_at_target == CIRCLE:
            possible_children.append(v)

    # Spouses: other parents of (definite + possible) children
    children_set = set(definite_children)
    if include_possible:
        children_set |= set(possible_children)

    spouses = set()
    for child in children_set:
        for v in range(n):
            if v == target or v == child:
                continue
            mark_at_child = pag_np[v, child]
            mark_at_v_from_child = pag_np[child, v]

            # v -> child (definite parent)
            is_parent = (mark_at_child == ARROW and mark_at_v_from_child == TAIL)
            # v o-> child (possible parent)
            is_possible_parent = (
                mark_at_child == ARROW and mark_at_v_from_child == CIRCLE
            )
            # v <-> child (bidirected)
            is_bidirected = (
                mark_at_child == ARROW and mark_at_v_from_child == ARROW
            )

            if is_parent or (include_possible and is_possible_parent) or is_bidirected:
                if v not in definite_parents and v not in possible_parents:
                    spouses.add(v)

    # Assemble full MB
    mb_set = {target}
    mb_set.update(definite_parents)
    mb_set.update(definite_children)
    mb_set.update(bidirected)
    mb_set.update(spouses)
    if include_possible:
        mb_set.update(possible_parents)
        mb_set.update(possible_children)

    mb_all = sorted(mb_set)

    return {
        'definite_parents': sorted(definite_parents),
        'definite_children': sorted(definite_children),
        'possible_parents': sorted(possible_parents),
        'possible_children': sorted(possible_children),
        'spouses': sorted(spouses),
        'bidirected': sorted(bidirected),
        'mb_all': mb_all,
        'mb_size': len(mb_all),
    }


def print_pag(pag: np.ndarray, var_names: Optional[List[str]] = None):
    """
    Print PAG edges in human-readable format.

    Args:
        pag: (n_vars, n_vars) PAG matrix
        var_names: optional variable names (default: X0, X1, ...)
    """
    pag_np = np.array(pag)
    n = pag_np.shape[0]

    if var_names is None:
        var_names = [f"X{i}" for i in range(n)]

    edge_symbols = {
        (ARROW, TAIL):   '-->',
        (TAIL, ARROW):   '<--',
        (ARROW, ARROW):  '<->',
        (ARROW, CIRCLE): 'o->',
        (CIRCLE, ARROW): '<-o',
        (CIRCLE, CIRCLE): 'o-o',
        (TAIL, CIRCLE):  '--o',
        (CIRCLE, TAIL):  'o--',
        (TAIL, TAIL):    '---',
    }

    print("PAG edges:")
    for i in range(n):
        for j in range(i + 1, n):
            if pag_np[i, j] == NONE and pag_np[j, i] == NONE:
                continue
            mi = pag_np[i, j]  # mark at j
            mj = pag_np[j, i]  # mark at i

            symbol = edge_symbols.get((mi, mj), '???')
            # Flip symbol direction for display: show mark_at_i --- mark_at_j
            # pag[j,i] = mark at i, pag[i,j] = mark at j
            left_mark = _MARK_NAMES.get(mj, '?')
            right_mark = _MARK_NAMES.get(mi, '?')
            print(f"  {var_names[i]} {left_mark}--{right_mark} {var_names[j]}")
