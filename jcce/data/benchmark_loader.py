"""
Unified Benchmark Dataset Loader (v16)

Loads and prepares datasets for JCCE experiments:
- LUCAS (synthetic, ground truth DAG + MB available)
- Sachs (protein signaling, ground truth DAG available)
- Asia (BnLearn, ground truth DAG available)
- CHILD-Gaussian (Gaussian SEM on expert DAG topology, d=19, ground truth)
- Neuropathic Pain (binary subgraph from Tu et al. 2019, d=28, ground truth)
- Diabetes (Pima, no ground truth)
- Heart Disease (UCI, no ground truth)
- Breast Cancer (Wisconsin, no ground truth)

Each dataset returns:
- X: feature matrix (n_samples, n_features)
- Y: target vector (n_samples,)
- config: dict with metadata (feature_names, true_mb, true_dag, treatment info, etc.)

Usage:
    from jcce.data.benchmark_loader import load_dataset, list_datasets

    X, Y, config = load_dataset('lucas')
    X, Y, config = load_dataset('child')
    X, Y, config = load_dataset('neuropathic_pain')
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# Ground Truth DAGs
# =============================================================================


def _build_lucas_true_dag() -> np.ndarray:
    """
    LUCAS true causal graph (12x12, node 11 = Lung_Cancer).
    Source: https://www.causality.inf.ethz.ch/data/LUCAS.html

    12 true edges:
        Anxiety(2) -> Smoking(0), Peer_Pressure(3) -> Smoking(0),
        Smoking(0) -> Yellow_Fingers(1), Smoking(0) -> Lung_Cancer(11),
        Genetics(4) -> Lung_Cancer(11), Genetics(4) -> Attention_Disorder(5),
        Allergy(9) -> Coughing(10), Lung_Cancer(11) -> Coughing(10),
        Lung_Cancer(11) -> Fatigue(8), Coughing(10) -> Fatigue(8),
        Attention_Disorder(5) -> Car_Accident(7), Fatigue(8) -> Car_Accident(7)
    """
    A = np.zeros((12, 12))
    A[2, 0] = 1  # Anxiety -> Smoking
    A[3, 0] = 1  # Peer_Pressure -> Smoking
    A[0, 1] = 1  # Smoking -> Yellow_Fingers
    A[0, 11] = 1  # Smoking -> Lung_Cancer
    A[4, 11] = 1  # Genetics -> Lung_Cancer
    A[4, 5] = 1  # Genetics -> Attention_Disorder
    A[9, 10] = 1  # Allergy -> Coughing
    A[11, 10] = 1  # Lung_Cancer -> Coughing
    A[11, 8] = 1  # Lung_Cancer -> Fatigue
    A[10, 8] = 1  # Coughing -> Fatigue
    A[5, 7] = 1  # Attention_Disorder -> Car_Accident
    A[8, 7] = 1  # Fatigue -> Car_Accident
    return A


def _build_sachs_true_dag() -> np.ndarray:
    """
    Sachs protein signaling network (11 nodes, 17 edges).
    Source: Sachs et al. (2005) "Causal Protein-Signaling Networks"

    Nodes: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, JNK
    """
    A = np.zeros((11, 11))
    # Node indices: 0=Raf, 1=Mek, 2=Plcg, 3=PIP2, 4=PIP3, 5=Erk, 6=Akt, 7=PKA, 8=PKC, 9=P38, 10=JNK
    A[0, 1] = 1  # Raf -> Mek
    A[1, 5] = 1  # Mek -> Erk
    A[2, 3] = 1  # Plcg -> PIP2
    A[2, 4] = 1  # Plcg -> PIP3
    A[4, 3] = 1  # PIP3 -> PIP2
    A[5, 6] = 1  # Erk -> Akt
    A[7, 0] = 1  # PKA -> Raf
    A[7, 1] = 1  # PKA -> Mek
    A[7, 5] = 1  # PKA -> Erk
    A[7, 6] = 1  # PKA -> Akt
    A[7, 9] = 1  # PKA -> P38
    A[7, 10] = 1  # PKA -> JNK
    A[8, 0] = 1  # PKC -> Raf
    A[8, 1] = 1  # PKC -> Mek
    A[8, 9] = 1  # PKC -> P38
    A[8, 10] = 1  # PKC -> JNK
    A[8, 2] = 1  # PKC -> Plcg
    return A


def _build_asia_true_dag() -> np.ndarray:
    """
    Asia (Lauritzen & Spiegelhalter, 1988) Bayesian network.
    8 nodes, 8 edges.

    Nodes: 0=asia, 1=tub, 2=smoke, 3=lung, 4=bronc, 5=either, 6=xray, 7=dysp
    """
    A = np.zeros((8, 8))
    A[0, 1] = 1  # asia -> tub
    A[2, 3] = 1  # smoke -> lung
    A[2, 4] = 1  # smoke -> bronc
    A[1, 5] = 1  # tub -> either
    A[3, 5] = 1  # lung -> either
    A[5, 6] = 1  # either -> xray
    A[5, 7] = 1  # either -> dysp
    A[4, 7] = 1  # bronc -> dysp
    return A


def _build_alarm_true_dag() -> np.ndarray:
    """
    ALARM network (Beinlich et al., 1989) — ICU patient monitoring.
    37 nodes, 46 edges.

    Nodes (BIF order from bnlearn.com/bnrepository):
        0=HISTORY, 1=CVP, 2=PCWP, 3=HYPOVOLEMIA, 4=LVEDVOLUME,
        5=LVFAILURE, 6=STROKEVOLUME, 7=ERRLOWOUTPUT, 8=HRBP,
        9=HREKG, 10=ERRCAUTER, 11=HRSAT, 12=INSUFFANESTH,
        13=ANAPHYLAXIS, 14=TPR, 15=EXPCO2, 16=KINKEDTUBE,
        17=MINVOL, 18=FIO2, 19=PVSAT, 20=SAO2, 21=PAP,
        22=PULMEMBOLUS, 23=SHUNT, 24=INTUBATION, 25=PRESS,
        26=DISCONNECT, 27=MINVOLSET, 28=VENTMACH, 29=VENTTUBE,
        30=VENTLUNG, 31=VENTALV, 32=ARTCO2, 33=CATECHOL, 34=HR,
        35=CO, 36=BP

    Source: bnlearn.com/bnrepository/alarm/alarm.bif.gz (verified)
    """
    A = np.zeros((37, 37))
    edges = [
        (5, 0),  # LVFAILURE -> HISTORY
        (4, 1),  # LVEDVOLUME -> CVP
        (4, 2),  # LVEDVOLUME -> PCWP
        (3, 4),  # HYPOVOLEMIA -> LVEDVOLUME
        (5, 4),  # LVFAILURE -> LVEDVOLUME
        (3, 6),  # HYPOVOLEMIA -> STROKEVOLUME
        (5, 6),  # LVFAILURE -> STROKEVOLUME
        (7, 8),  # ERRLOWOUTPUT -> HRBP
        (34, 8),  # HR -> HRBP
        (10, 9),  # ERRCAUTER -> HREKG
        (34, 9),  # HR -> HREKG
        (10, 11),  # ERRCAUTER -> HRSAT
        (34, 11),  # HR -> HRSAT
        (13, 14),  # ANAPHYLAXIS -> TPR
        (32, 15),  # ARTCO2 -> EXPCO2
        (30, 15),  # VENTLUNG -> EXPCO2
        (24, 17),  # INTUBATION -> MINVOL
        (30, 17),  # VENTLUNG -> MINVOL
        (18, 19),  # FIO2 -> PVSAT
        (31, 19),  # VENTALV -> PVSAT
        (19, 20),  # PVSAT -> SAO2
        (23, 20),  # SHUNT -> SAO2
        (22, 21),  # PULMEMBOLUS -> PAP
        (24, 23),  # INTUBATION -> SHUNT
        (22, 23),  # PULMEMBOLUS -> SHUNT
        (24, 25),  # INTUBATION -> PRESS
        (16, 25),  # KINKEDTUBE -> PRESS
        (29, 25),  # VENTTUBE -> PRESS
        (27, 28),  # MINVOLSET -> VENTMACH
        (26, 29),  # DISCONNECT -> VENTTUBE
        (28, 29),  # VENTMACH -> VENTTUBE
        (24, 30),  # INTUBATION -> VENTLUNG
        (16, 30),  # KINKEDTUBE -> VENTLUNG
        (29, 30),  # VENTTUBE -> VENTLUNG
        (24, 31),  # INTUBATION -> VENTALV
        (30, 31),  # VENTLUNG -> VENTALV
        (31, 32),  # VENTALV -> ARTCO2
        (32, 33),  # ARTCO2 -> CATECHOL
        (12, 33),  # INSUFFANESTH -> CATECHOL
        (20, 33),  # SAO2 -> CATECHOL
        (14, 33),  # TPR -> CATECHOL
        (33, 34),  # CATECHOL -> HR
        (34, 35),  # HR -> CO
        (6, 35),  # STROKEVOLUME -> CO
        (35, 36),  # CO -> BP
        (14, 36),  # TPR -> BP
    ]
    for i, j in edges:
        A[i, j] = 1
    return A


def _build_insurance_true_dag() -> np.ndarray:
    """
    INSURANCE network — car insurance risk evaluation.
    27 nodes, 52 edges.

    Nodes (BIF order from bnlearn.com/bnrepository):
        0=GoodStudent, 1=Age, 2=SocioEcon, 3=RiskAversion,
        4=VehicleYear, 5=ThisCarDam, 6=RuggedAuto, 7=Accident,
        8=MakeModel, 9=DrivQuality, 10=Mileage, 11=Antilock,
        12=DrivingSkill, 13=SeniorTrain, 14=ThisCarCost, 15=Theft,
        16=CarValue, 17=HomeBase, 18=AntiTheft, 19=PropCost,
        20=OtherCarCost, 21=OtherCar, 22=MedCost, 23=Cushioning,
        24=Airbag, 25=ILiCost, 26=DrivHist

    Source: bnlearn.com/bnrepository/insurance/insurance.bif.gz (verified)
    """
    A = np.zeros((27, 27))
    edges = [
        (2, 0),  # SocioEcon -> GoodStudent
        (1, 0),  # Age -> GoodStudent
        (1, 2),  # Age -> SocioEcon
        (1, 3),  # Age -> RiskAversion
        (2, 3),  # SocioEcon -> RiskAversion
        (2, 4),  # SocioEcon -> VehicleYear
        (3, 4),  # RiskAversion -> VehicleYear
        (7, 5),  # Accident -> ThisCarDam
        (6, 5),  # RuggedAuto -> ThisCarDam
        (8, 6),  # MakeModel -> RuggedAuto
        (4, 6),  # VehicleYear -> RuggedAuto
        (11, 7),  # Antilock -> Accident
        (10, 7),  # Mileage -> Accident
        (9, 7),  # DrivQuality -> Accident
        (2, 8),  # SocioEcon -> MakeModel
        (3, 8),  # RiskAversion -> MakeModel
        (12, 9),  # DrivingSkill -> DrivQuality
        (3, 9),  # RiskAversion -> DrivQuality
        (8, 11),  # MakeModel -> Antilock
        (4, 11),  # VehicleYear -> Antilock
        (1, 12),  # Age -> DrivingSkill
        (13, 12),  # SeniorTrain -> DrivingSkill
        (1, 13),  # Age -> SeniorTrain
        (3, 13),  # RiskAversion -> SeniorTrain
        (5, 14),  # ThisCarDam -> ThisCarCost
        (16, 14),  # CarValue -> ThisCarCost
        (15, 14),  # Theft -> ThisCarCost
        (18, 15),  # AntiTheft -> Theft
        (17, 15),  # HomeBase -> Theft
        (16, 15),  # CarValue -> Theft
        (8, 16),  # MakeModel -> CarValue
        (4, 16),  # VehicleYear -> CarValue
        (10, 16),  # Mileage -> CarValue
        (3, 17),  # RiskAversion -> HomeBase
        (2, 17),  # SocioEcon -> HomeBase
        (3, 18),  # RiskAversion -> AntiTheft
        (2, 18),  # SocioEcon -> AntiTheft
        (20, 19),  # OtherCarCost -> PropCost
        (14, 19),  # ThisCarCost -> PropCost
        (7, 20),  # Accident -> OtherCarCost
        (6, 20),  # RuggedAuto -> OtherCarCost
        (2, 21),  # SocioEcon -> OtherCar
        (7, 22),  # Accident -> MedCost
        (1, 22),  # Age -> MedCost
        (23, 22),  # Cushioning -> MedCost
        (6, 23),  # RuggedAuto -> Cushioning
        (24, 23),  # Airbag -> Cushioning
        (8, 24),  # MakeModel -> Airbag
        (4, 24),  # VehicleYear -> Airbag
        (7, 25),  # Accident -> ILiCost
        (12, 26),  # DrivingSkill -> DrivHist
        (3, 26),  # RiskAversion -> DrivHist
    ]
    for i, j in edges:
        A[i, j] = 1
    return A


def _build_child_true_dag() -> np.ndarray:
    """
    CHILD Bayesian network (Spiegelhalter & Cowell, 1992).
    20 nodes, 25 edges. Congenital heart disease diagnosis.

    Nodes (0-indexed):
        0=BirthAsphyxia, 1=Disease, 2=Age, 3=Sick, 4=LVH,
        5=DuctFlow, 6=CardiacMixing, 7=LungParench, 8=LungFlow,
        9=Grunting, 10=HypDistrib, 11=HypoxiaInO2, 12=CO2,
        13=ChestXray, 14=LowerBodyO2, 15=RUQO2, 16=CO2Report,
        17=XrayReport, 18=GruntingReport, 19=LVHreport

    Source: bnlearn.com/bnrepository (child.bif)
    """
    A = np.zeros((20, 20))
    edges = [
        (0, 1),  # BirthAsphyxia -> Disease
        (1, 2),  # Disease -> Age
        (1, 3),  # Disease -> Sick
        (1, 4),  # Disease -> LVH
        (1, 5),  # Disease -> DuctFlow
        (1, 6),  # Disease -> CardiacMixing
        (1, 7),  # Disease -> LungParench
        (1, 8),  # Disease -> LungFlow
        (3, 2),  # Sick -> Age
        (3, 9),  # Sick -> Grunting
        (7, 9),  # LungParench -> Grunting
        (7, 11),  # LungParench -> HypoxiaInO2
        (7, 12),  # LungParench -> CO2
        (7, 13),  # LungParench -> ChestXray
        (8, 13),  # LungFlow -> ChestXray
        (5, 10),  # DuctFlow -> HypDistrib
        (6, 10),  # CardiacMixing -> HypDistrib
        (6, 11),  # CardiacMixing -> HypoxiaInO2
        (10, 14),  # HypDistrib -> LowerBodyO2
        (11, 14),  # HypoxiaInO2 -> LowerBodyO2
        (11, 15),  # HypoxiaInO2 -> RUQO2
        (4, 19),  # LVH -> LVHreport
        (12, 16),  # CO2 -> CO2Report
        (13, 17),  # ChestXray -> XrayReport
        (9, 18),  # Grunting -> GruntingReport
    ]
    for i, j in edges:
        A[i, j] = 1
    return A


def _build_neuropathic_pain_subgraph_dag() -> Tuple[np.ndarray, List[str], int]:
    """
    Extract a causally-closed subgraph from the Neuropathic Pain Diagnosis
    Simulator (Tu et al., NeurIPS 2019). 222-node BN → 29-node subgraph.

    Target: DLI L4-L5 (disc ligament injury at lumbar level 4-5).
    Features: 2 radiculopathies + 26 symptoms downstream of this DLI.

    The full 222-node DAG has three layers with no within-layer edges:
        Layer 1 (0-26): 27 disc ligament injuries (DLI)
        Layer 2 (27-78): 52 radiculopathies (left/right per nerve root)
        Layer 3 (79-221): 143 symptoms

    Source: github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator

    Returns:
        sub_dag: (29, 29) adjacency matrix of the subgraph
        node_names: List of 29 node names
        target_pos: Index of the target node (DLI) in the subgraph (always 0)
    """
    # Load the full CPDAG and convert to DAG
    cpdag_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "benchmarks"
        / "neuropathic_pain"
        / "cpdag_GT.csv"
    )
    if not cpdag_path.exists():
        raise FileNotFoundError(
            f"Neuropathic Pain CPDAG not found at {cpdag_path}. "
            f"Download cpdag_GT.csv from: "
            f"https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator"
        )

    import pandas as pd

    df = pd.read_csv(cpdag_path, index_col=0)
    A = df.values.astype(int)

    # Convert CPDAG to DAG: directed edges keep direction;
    # bidirectional edges (Markov equivalence) oriented lower→higher index
    # (consistent with the layered structure: DLI→Radi→Symptom)
    dag = np.zeros((222, 222), dtype=int)
    for i in range(222):
        for j in range(222):
            if A[i, j] == 1 and A[j, i] == 0:
                dag[i, j] = 1
            elif A[i, j] == 1 and A[j, i] == 1 and i < j:
                dag[i, j] = 1

    # Extract subgraph: DLI 10 (L4-L5) → its radiculopathies → their symptoms
    dli_idx = 10
    radi_children = list(np.where(dag[dli_idx, 27:79] == 1)[0] + 27)
    symptom_children = set()
    for r in radi_children:
        syms = np.where(dag[r, 79:] == 1)[0] + 79
        symptom_children.update(syms.tolist())

    all_nodes = sorted([dli_idx] + radi_children + sorted(symptom_children))
    sub_dag = dag[np.ix_(all_nodes, all_nodes)]
    target_pos = all_nodes.index(dli_idx)

    # Build node names
    node_names = []
    for idx in all_nodes:
        if idx < 27:
            node_names.append("DLI_L4L5")
        elif idx < 79:
            radi_num = idx - 27
            side = "L" if radi_num % 2 == 0 else "R"
            root_level = radi_num // 2 + 2
            node_names.append(f"{side}_Radi_L{root_level}")
        else:
            node_names.append(f"Symptom_{idx - 79}")

    return sub_dag, node_names, target_pos


def _sample_neuropathic_pain_bn(
    dag: np.ndarray,
    n_samples: int = 5000,
    seed: int = 42,
    base_rate: float = 0.15,
    leak_prob: float = 0.02,
) -> np.ndarray:
    """
    Sample binary data from a noisy-OR model over the subgraph DAG.

    The original BN CPDs were estimated from 141 patients, but require
    pgmpy 0.1.7 to load. Instead, we use a noisy-OR model which is the
    standard generative model for binary diagnostic BNs (Pearl, 1988).

    For each node X with parents Pa(X):
        P(X=0 | Pa(X)) = (1 - leak) * prod_{j in Pa(X), Pa_j=1} (1 - w_j)
        P(X=1 | Pa(X)) = 1 - P(X=0 | Pa(X))

    Edge weights w_j are drawn once from Beta(2, 3) (mean ~0.4).

    Args:
        dag: (d, d) binary adjacency matrix
        n_samples: Number of samples
        seed: Random seed
        base_rate: Prior probability for root nodes
        leak_prob: Leak probability (spontaneous activation)

    Returns:
        data: (n_samples, d) binary array
    """
    rng = np.random.RandomState(seed)
    d = dag.shape[0]

    # Generate fixed edge weights from Beta(2, 3), mean ~0.4
    weight_rng = np.random.RandomState(seed + 1000)
    edge_weights = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if dag[i, j] == 1:
                edge_weights[i, j] = weight_rng.beta(2, 3) * 0.8 + 0.1  # range [0.1, 0.9]

    # Topological order (layered graph: nodes with no parents first)
    in_degree = dag.sum(axis=0)
    topo_order = np.argsort(in_degree)  # 0-in-degree first

    data = np.zeros((n_samples, d), dtype=np.float32)
    for n in range(n_samples):
        sample = np.zeros(d)
        for node in topo_order:
            parents = np.where(dag[:, node] == 1)[0]
            if len(parents) == 0:
                # Root node: sample from base rate
                sample[node] = 1.0 if rng.random() < base_rate else 0.0
            else:
                # Noisy-OR: P(X=0) = (1-leak) * prod_{active parents} (1 - w_j)
                p_off = 1.0 - leak_prob
                for p in parents:
                    if sample[p] == 1.0:
                        p_off *= 1.0 - edge_weights[p, node]
                p_on = 1.0 - p_off
                sample[node] = 1.0 if rng.random() < p_on else 0.0
        data[n] = sample

    return data


def _generate_child_gaussian_sem(
    dag: np.ndarray,
    n_samples: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate continuous data from the CHILD DAG using a Linear Gaussian SEM.

    Standard approach from NOTEARS/GOLEM/DAGMA papers:
        X_j = sum_{i in Pa(j)} beta_{ij} * X_i + eps_j
        beta ~ Uniform([-2,-0.5] U [0.5,2])
        eps ~ N(0, 1)

    The target (BirthAsphyxia, node 0) is binarized via logistic threshold
    and excluded from the feature matrix.

    Args:
        dag: (20, 20) binary adjacency matrix (CHILD network)
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        X: (n_samples, 19) continuous feature matrix (nodes 1-19)
        Y: (n_samples,) binary target (BirthAsphyxia binarized)
    """
    rng = np.random.RandomState(seed)
    d = dag.shape[0]

    # Generate edge weights: Uniform([-2,-0.5] U [0.5,2])
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if dag[i, j] == 1:
                w = rng.uniform(0.5, 2.0)
                if rng.random() < 0.5:
                    w = -w
                W[i, j] = w

    # Topological order
    in_degree = dag.sum(axis=0).astype(int)
    remaining = set(range(d))
    topo_order = []
    while remaining:
        for node in sorted(remaining):
            parents_remaining = [p for p in range(d) if dag[p, node] == 1 and p in remaining]
            if not parents_remaining:
                topo_order.append(node)
                remaining.discard(node)
                break

    # Generate data in topological order
    X = np.zeros((n_samples, d), dtype=np.float64)
    for node in topo_order:
        parents = np.where(dag[:, node] == 1)[0]
        noise = rng.randn(n_samples)
        if len(parents) > 0:
            X[:, node] = X[:, parents] @ W[parents, node] + noise
        else:
            X[:, node] = noise

    # BirthAsphyxia (node 0) is the root → binarize as target
    # Use median split (node 0 is purely noise, so this gives ~50% balance)
    target_col = 0
    median_val = np.median(X[:, target_col])
    Y = (X[:, target_col] > median_val).astype(np.float32)

    # Feature matrix: all nodes except target
    feature_cols = [i for i in range(d) if i != target_col]
    X_features = X[:, feature_cols].astype(np.float32)

    return X_features, Y


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    "lucas": {
        "name": "LUCAS",
        "description": "Synthetic causal dataset with known ground truth (lung cancer)",
        "task": "classification",
        "n_expected": 2000,
        "p_expected": 11,
        "feature_names": [
            "Smoking",
            "Yellow_Fingers",
            "Anxiety",
            "Peer_Pressure",
            "Genetics",
            "Attention_Disorder",
            "Born_an_Even_Day",
            "Car_Accident",
            "Fatigue",
            "Allergy",
            "Coughing",
        ],
        "target_name": "Lung_cancer",
        "true_mb": [0, 4, 8, 9, 10],  # Smoking, Genetics, Fatigue, Allergy, Coughing
        "has_true_dag": True,
        "treatment_idx": 0,
        "treatment_name": "Smoking",
        "immutable_features": [],  # All features are binary/synthetic, none immutable
        "categorical_features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # All 11 features are binary
        # NSGA-II defaults
        "pop_size": 30,
        "n_gen": 40,
        "max_iter": 150,
        # Quick test config
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "sachs": {
        "name": "Sachs (Protein Signaling)",
        "description": "Flow cytometry protein signaling (Sachs et al. 2005)",
        "task": "classification",  # Can binarize Erk or Akt as outcome
        "n_expected": 7466,
        "p_expected": 11,
        "feature_names": [
            "Raf",
            "Mek",
            "Plcg",
            "PIP2",
            "PIP3",
            "Erk",
            "Akt",
            "PKA",
            "PKC",
            "P38",
            "JNK",
        ],
        "target_name": "Erk",  # Default: binarize Erk (above/below median)
        "target_idx": 5,  # Index in original data
        "true_mb": None,  # Derived from DAG at load time
        "has_true_dag": True,
        "treatment_idx": 7,  # PKA (known activator)
        "treatment_name": "PKA",
        "pop_size": 30,
        "n_gen": 40,
        "max_iter": 150,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "asia": {
        "name": "Asia (Lauritzen-Spiegelhalter)",
        "description": "Classic BnLearn benchmark (chest clinic diagnosis)",
        "task": "classification",
        "n_expected": 5000,  # Will be sampled
        "p_expected": 7,  # 8 nodes minus target
        "feature_names": ["asia", "tub", "smoke", "lung", "bronc", "either", "xray"],
        "target_name": "dysp",  # Dyspnea as outcome
        "target_idx": 7,
        "true_mb": [4, 5],  # bronc, either → dysp
        "has_true_dag": True,
        "treatment_idx": 2,  # smoke
        "treatment_name": "smoke",
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 100,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "diabetes": {
        "name": "Diabetes (Pima Indians)",
        "description": "UCI Pima Indians diabetes classification",
        "task": "classification",
        "n_expected": 768,
        "p_expected": 8,
        "feature_names": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "target_name": "Outcome",
        "true_mb": None,
        "has_true_dag": False,
        "treatment_idx": 1,
        "treatment_name": "Glucose",
        "zero_as_missing_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "standardize": True,
        "immutable_features": [0, 7],  # Pregnancies, Age — cannot be changed
        "categorical_features": [],  # All continuous after standardization
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 100,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "heart_disease": {
        "name": "Heart Disease (Cleveland)",
        "description": "UCI Cleveland heart disease classification",
        "task": "classification",
        "n_expected": 297,
        "p_expected": 13,
        "feature_names": None,  # Loaded from file
        "target_name": "target",
        "true_mb": None,
        "has_true_dag": False,
        "treatment_idx": None,
        "treatment_name": None,
        "immutable_features": [0, 1],  # age, sex — cannot be changed
        "categorical_features": [
            1,
            2,
            5,
            6,
            8,
            10,
            12,
        ],  # sex, cp, fbs, restecg, exang, slope, thal
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 100,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "breast_cancer": {
        "name": "Breast Cancer (Wisconsin)",
        "description": "UCI Wisconsin breast cancer diagnosis",
        "task": "classification",
        "n_expected": 569,
        "p_expected": 30,
        "feature_names": None,  # Loaded from file
        "target_name": "diagnosis",
        "true_mb": None,
        "has_true_dag": False,
        "treatment_idx": None,
        "treatment_name": None,
        "skip_dml": False,  # Auto-discover treatments from learned DAG
        "immutable_features": [],  # All features are cell measurements, none immutable
        "categorical_features": [],  # All 30 features are continuous measurements
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 100,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "child": {
        "name": "CHILD-Gaussian (Congenital Heart Disease)",
        "description": "Linear Gaussian SEM on CHILD BN topology (Spiegelhalter & Cowell 1992)",
        "task": "classification",
        "n_expected": 2000,
        "p_expected": 19,  # 20 nodes minus target
        "feature_names": [
            "Disease",
            "Age",
            "Sick",
            "LVH",
            "DuctFlow",
            "CardiacMixing",
            "LungParench",
            "LungFlow",
            "Grunting",
            "HypDistrib",
            "HypoxiaInO2",
            "CO2",
            "ChestXray",
            "LowerBodyO2",
            "RUQO2",
            "CO2Report",
            "XrayReport",
            "GruntingReport",
            "LVHreport",
        ],
        "target_name": "BirthAsphyxia",
        "target_idx": 0,  # Node 0 in the 20-node DAG
        "true_mb": [
            0
        ],  # Disease is the only child of BirthAsphyxia (index 0 in features = Disease)
        "has_true_dag": True,
        "treatment_idx": 0,  # Disease (main hub node)
        "treatment_name": "Disease",
        "categorical_features": [],  # All continuous (Gaussian SEM)
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 150,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "alarm": {
        "name": "ALARM-Gaussian (ICU Monitoring)",
        "description": "Linear Gaussian SEM on ALARM BN topology (Beinlich et al. 1989)",
        "task": "classification",
        "n_expected": 2000,
        "p_expected": 36,  # 37 nodes minus target
        "feature_names": [
            "HISTORY",
            "CVP",
            "PCWP",
            "HYPOVOLEMIA",
            "LVEDVOLUME",
            "STROKEVOLUME",
            "ERRLOWOUTPUT",
            "HRBP",
            "HREKG",
            "ERRCAUTER",
            "HRSAT",
            "INSUFFANESTH",
            "ANAPHYLAXIS",
            "TPR",
            "EXPCO2",
            "KINKEDTUBE",
            "MINVOL",
            "FIO2",
            "PVSAT",
            "SAO2",
            "PAP",
            "PULMEMBOLUS",
            "SHUNT",
            "INTUBATION",
            "PRESS",
            "DISCONNECT",
            "MINVOLSET",
            "VENTMACH",
            "VENTTUBE",
            "VENTLUNG",
            "VENTALV",
            "ARTCO2",
            "CATECHOL",
            "HR",
            "CO",
            "BP",
        ],
        "target_name": "LVFAILURE",
        "target_idx": 5,  # LVFAILURE is node 5 in the 37-node DAG
        "true_mb": None,  # Derived at load time
        "has_true_dag": True,
        "treatment_idx": 3,  # HYPOVOLEMIA
        "treatment_name": "HYPOVOLEMIA",
        "categorical_features": [],  # All continuous (Gaussian SEM)
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 150,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "insurance": {
        "name": "INSURANCE-Gaussian (Car Insurance Risk)",
        "description": "Linear Gaussian SEM on INSURANCE BN topology (expert-designed)",
        "task": "classification",
        "n_expected": 2000,
        "p_expected": 26,  # 27 nodes minus target
        "feature_names": [
            "GoodStudent",
            "Age",
            "SocioEcon",
            "RiskAversion",
            "VehicleYear",
            "ThisCarDam",
            "RuggedAuto",
            "Accident",
            "MakeModel",
            "DrivQuality",
            "Mileage",
            "Antilock",
            "DrivingSkill",
            "SeniorTrain",
            "ThisCarCost",
            "CarValue",
            "HomeBase",
            "AntiTheft",
            "PropCost",
            "OtherCarCost",
            "OtherCar",
            "MedCost",
            "Cushioning",
            "Airbag",
            "ILiCost",
            "DrivHist",
        ],
        "target_name": "Theft",
        "target_idx": 15,  # Theft is node 15 in the 27-node DAG (binary: True/False)
        "true_mb": None,  # Derived at load time
        "has_true_dag": True,
        "treatment_idx": 17,  # HomeBase (mapped to index in features after target removal)
        "treatment_name": "HomeBase",
        "categorical_features": [],  # All continuous (Gaussian SEM)
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 150,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
    "neuropathic_pain": {
        "name": "Neuropathic Pain (L4-L5 Subgraph)",
        "description": "Subgraph of Tu et al. 2019 NeurIPS simulator (DLI L4-L5 diagnosis)",
        "task": "classification",
        "n_expected": 5000,
        "p_expected": 28,  # 29 nodes minus target
        "feature_names": None,  # Set at load time from subgraph extraction
        "target_name": "DLI_L4L5",
        "target_idx": 0,  # Target is always first in the subgraph
        "true_mb": [1, 2],  # L_Radi and R_Radi are the only children of DLI
        "has_true_dag": True,
        "treatment_idx": 1,  # Left radiculopathy
        "treatment_name": "L_Radi_L23",
        "categorical_features": list(range(28)),  # All binary
        "pop_size": 20,
        "n_gen": 30,
        "max_iter": 150,
        "quick_pop_size": 6,
        "quick_n_gen": 3,
        "quick_max_iter": 30,
    },
}


# =============================================================================
# Dataset Loading Functions
# =============================================================================


def _load_lucas() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load LUCAS dataset from CSV."""
    csv_path = DATA_DIR / "benchmarks" / "lucas" / "raw" / "lucas0_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LUCAS data not found at {csv_path}. "
            f"Download from: https://www.causality.inf.ethz.ch/data/LUCAS.html"
        )
    df = pd.read_csv(csv_path)
    X = df.drop("Lung_cancer", axis=1).values.astype(np.float32)
    Y = df["Lung_cancer"].values.astype(np.float32)
    config = {**DATASET_CONFIGS["lucas"]}
    config["true_dag"] = _build_lucas_true_dag()
    return X, Y, config


def _load_sachs() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Sachs protein signaling dataset.

    If raw data exists, loads from CSV. Otherwise generates synthetic data
    from the known DAG structure for testing.
    """
    raw_dir = DATA_DIR / "benchmarks" / "sachs" / "raw"
    config = {**DATASET_CONFIGS["sachs"]}
    config["true_dag"] = _build_sachs_true_dag()

    # Try loading real data
    csv_candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xls"))
    if csv_candidates:
        df = (
            pd.read_csv(csv_candidates[0])
            if str(csv_candidates[0]).endswith(".csv")
            else pd.read_excel(csv_candidates[0])
        )
        # Sachs data has 11 proteins as columns
        target_col = config["target_name"]
        if target_col in df.columns:
            # Binarize target (above median = 1)
            median_val = df[target_col].median()
            Y = (df[target_col] > median_val).astype(np.float32)
            X = df.drop(target_col, axis=1).values.astype(np.float32)
        else:
            # Use last column as target
            Y = (df.iloc[:, -1] > df.iloc[:, -1].median()).astype(np.float32)
            X = df.iloc[:, :-1].values.astype(np.float32)
        config["feature_names"] = [c for c in df.columns if c != target_col]
    else:
        # Generate synthetic data from known DAG for testing
        warnings.warn(
            "Sachs raw data not found. Generating synthetic data from known DAG. "
            "Download real data from: https://www.bnlearn.com/bnrepository/discrete-medium.html#sachs"
        )
        X, Y = _generate_from_dag(config["true_dag"], n_samples=5000, target_idx=5)
        config["synthetic"] = True

    # Derive true MB from DAG
    dag = config["true_dag"]
    target_idx = config["target_idx"]
    mb = set()
    # Parents of target
    for i in range(dag.shape[0]):
        if dag[i, target_idx] > 0:
            mb.add(i)
    # Children of target
    for j in range(dag.shape[1]):
        if dag[target_idx, j] > 0:
            mb.add(j)
    # Spouses (parents of children)
    children = {j for j in range(dag.shape[1]) if dag[target_idx, j] > 0}
    for child in children:
        for i in range(dag.shape[0]):
            if dag[i, child] > 0 and i != target_idx:
                mb.add(i)
    mb.discard(target_idx)
    # Remap indices (target removed from X, so shift indices > target_idx)
    config["true_mb"] = sorted([i if i < target_idx else i - 1 for i in mb])

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, Y, config


def _sample_asia_bn(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """
    Sample from the exact Asia BN conditional probability tables.

    CPTs from Lauritzen & Spiegelhalter (1988), sourced from bnlearn BIF.
    Node order: asia, tub, smoke, lung, bronc, either, xray, dysp
    """
    rng = np.random.RandomState(seed)

    # Root nodes
    asia = rng.random(n_samples) < 0.01
    smoke = rng.random(n_samples) < 0.5

    # Conditional nodes
    tub = rng.random(n_samples) < np.where(asia, 0.05, 0.01)
    lung = rng.random(n_samples) < np.where(smoke, 0.1, 0.01)
    bronc = rng.random(n_samples) < np.where(smoke, 0.6, 0.3)

    # Either = lung OR tub (deterministic)
    either = lung | tub

    # Xray
    xray = rng.random(n_samples) < np.where(either, 0.98, 0.05)

    # Dyspnea (depends on bronc AND either)
    p_dysp = np.where(
        bronc & either, 0.9, np.where(~bronc & either, 0.7, np.where(bronc & ~either, 0.8, 0.1))
    )
    dysp = rng.random(n_samples) < p_dysp

    return np.column_stack([asia, tub, smoke, lung, bronc, either, xray, dysp]).astype(np.float32)


def _load_asia() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Asia dataset. Samples from exact BN CPTs if CSV not available.
    """
    config = {**DATASET_CONFIGS["asia"]}
    config["true_dag"] = _build_asia_true_dag()

    raw_dir = DATA_DIR / "benchmarks" / "asia"
    csv_path = raw_dir / "asia.csv" if raw_dir.exists() else None

    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        target_col = config["target_name"]
        Y = df[target_col].values.astype(np.float32)
        X = df.drop(target_col, axis=1).values.astype(np.float32)
        config["feature_names"] = [c for c in df.columns if c != target_col]
    else:
        data = _sample_asia_bn(n_samples=5000, seed=42)
        # Node order: asia(0), tub(1), smoke(2), lung(3), bronc(4), either(5), xray(6), dysp(7)
        Y = data[:, 7].astype(np.float32)
        X = data[:, :7].astype(np.float32)

    return X, Y, config


def _load_diabetes() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load Pima Indians Diabetes dataset."""
    config = {**DATASET_CONFIGS["diabetes"]}

    # Try multiple locations
    candidates = [
        DATA_DIR / "medical" / "diabetes.csv",
        DATA_DIR / "benchmarks" / "diabetes" / "diabetes.csv",
        PROJECT_ROOT / "data" / "diabetes.csv",
    ]

    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            break

    if df is None:
        # Try sklearn
        try:

            warnings.warn(
                "Loading Pima diabetes from local CSV failed. "
                "Using a placeholder. Download from Kaggle."
            )
            raise FileNotFoundError("Pima diabetes not in sklearn")
        except Exception:
            raise FileNotFoundError(
                "Diabetes data not found. Download from Kaggle: "
                "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
            )

    X = df.drop(config["target_name"], axis=1).values.astype(np.float32)
    Y = df[config["target_name"]].values.astype(np.float32)

    # Impute zero-as-missing
    if "zero_as_missing_cols" in config:
        for col_name in config["zero_as_missing_cols"]:
            col_idx = list(df.columns).index(col_name)
            mask = X[:, col_idx] == 0
            if mask.any():
                median_val = np.median(X[~mask, col_idx])
                X[mask, col_idx] = median_val

    # Standardize
    if config.get("standardize", False):
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    return X, Y, config


def _load_heart_disease() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load Heart Disease dataset."""
    config = {**DATASET_CONFIGS["heart_disease"]}

    npy_dir = DATA_DIR / "medical" / "heart_disease"
    if npy_dir.exists() and (npy_dir / "X_train.npy").exists():
        X_train = np.load(npy_dir / "X_train.npy")
        X_test = np.load(npy_dir / "X_test.npy")
        Y_train = np.load(npy_dir / "y_train.npy")
        Y_test = np.load(npy_dir / "y_test.npy")
        X = np.vstack([X_train, X_test]).astype(np.float32)
        Y = np.concatenate([Y_train, Y_test]).astype(np.float32)

        names_path = npy_dir / "feature_names.txt"
        if names_path.exists():
            config["feature_names"] = [line.strip() for line in open(names_path)]
    else:
        raise FileNotFoundError(
            f"Heart disease data not found at {npy_dir}. Run data preparation script first."
        )

    return X, Y, config


def _load_breast_cancer() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load Breast Cancer Wisconsin dataset."""
    config = {**DATASET_CONFIGS["breast_cancer"]}

    npy_dir = DATA_DIR / "medical" / "breast_cancer"
    if npy_dir.exists() and (npy_dir / "X_train.npy").exists():
        X_train = np.load(npy_dir / "X_train.npy")
        X_test = np.load(npy_dir / "X_test.npy")
        Y_train = np.load(npy_dir / "y_train.npy")
        Y_test = np.load(npy_dir / "y_test.npy")
        X = np.vstack([X_train, X_test]).astype(np.float32)
        Y = np.concatenate([Y_train, Y_test]).astype(np.float32)

        names_path = npy_dir / "feature_names.txt"
        if names_path.exists():
            config["feature_names"] = [line.strip() for line in open(names_path)]
    else:
        # Fallback: sklearn
        from sklearn.datasets import load_breast_cancer as sk_load

        data = sk_load()
        X = data.data.astype(np.float32)
        Y = data.target.astype(np.float32)
        config["feature_names"] = list(data.feature_names)

    return X, Y, config


def _generate_from_dag(
    true_dag: np.ndarray,
    n_samples: int = 5000,
    target_idx: int = -1,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic binary data from a known DAG structure.
    Uses linear SCM with logistic thresholding for binary variables.
    """
    rng = np.random.RandomState(seed)
    n_vars = true_dag.shape[0]

    # Topological sort
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for parent in range(n_vars):
            if true_dag[parent, node] > 0:
                dfs(parent)
        order.append(node)

    for i in range(n_vars):
        dfs(i)

    # Generate data in topological order
    data = np.zeros((n_samples, n_vars))
    weights = rng.uniform(0.5, 2.0, size=(n_vars, n_vars)) * true_dag

    for node in order:
        noise = rng.randn(n_samples) * noise_std
        parents = np.where(true_dag[:, node] > 0)[0]
        if len(parents) > 0:
            linear_combo = data[:, parents] @ weights[parents, node] + noise
        else:
            linear_combo = noise
        # Logistic thresholding for binary
        prob = 1.0 / (1.0 + np.exp(-linear_combo))
        data[:, node] = (prob > 0.5).astype(float)

    if target_idx == -1:
        target_idx = n_vars - 1

    Y = data[:, target_idx].astype(np.float32)
    X = np.delete(data, target_idx, axis=1).astype(np.float32)
    return X, Y


def _load_child() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load CHILD-Gaussian dataset.

    Generates continuous data from the CHILD BN topology (20 nodes, 25 edges)
    using a Linear Gaussian SEM. BirthAsphyxia (root node) is binarized as
    the classification target.

    This approach is standard in NOTEARS/GOLEM/DAGMA papers: use an expert-
    designed DAG topology with continuous SEM data generation. The DAG topology
    (evaluated by SHD/F1/SID) comes from the original expert network.
    """
    config = {**DATASET_CONFIGS["child"]}
    full_dag = _build_child_true_dag()
    config["true_dag"] = full_dag

    X, Y = _generate_child_gaussian_sem(full_dag, n_samples=2000, seed=42)

    return X, Y, config


def _load_neuropathic_pain() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load Neuropathic Pain subgraph dataset.

    Extracts a 29-node causally-closed subgraph from the 222-node Neuropathic
    Pain Diagnosis Simulator (Tu et al., NeurIPS 2019). Target is the DLI L4-L5
    injury (binary: present/absent). Features are downstream radiculopathies
    and symptoms.

    Data is sampled from a noisy-OR model over the subgraph, producing binary
    features consistent with the DAG structure.
    """
    config = {**DATASET_CONFIGS["neuropathic_pain"]}

    sub_dag, node_names, target_pos = _build_neuropathic_pain_subgraph_dag()
    config["true_dag"] = sub_dag
    config["feature_names"] = [n for i, n in enumerate(node_names) if i != target_pos]

    # Sample binary data using noisy-OR model
    data = _sample_neuropathic_pain_bn(sub_dag, n_samples=5000, seed=42)

    Y = data[:, target_pos].astype(np.float32)
    feature_cols = [i for i in range(sub_dag.shape[0]) if i != target_pos]
    X = data[:, feature_cols].astype(np.float32)

    return X, Y, config


def _load_alarm() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load ALARM-Gaussian dataset.

    Generates continuous data from the ALARM BN topology (37 nodes, 46 edges)
    using a Linear Gaussian SEM. LVFAILURE (binary root node) is binarized
    as the classification target.
    """
    config = {**DATASET_CONFIGS["alarm"]}
    full_dag = _build_alarm_true_dag()
    config["true_dag"] = full_dag

    target_idx = config["target_idx"]
    X, Y = _generate_child_gaussian_sem(full_dag, n_samples=2000, seed=42)
    # _generate_child_gaussian_sem always uses node 0 as target.
    # For ALARM, target is node 5 (LVFAILURE), so we need a generalized version.
    # Re-generate with the correct target:
    rng = np.random.RandomState(42)
    d = full_dag.shape[0]
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if full_dag[i, j] == 1:
                w = rng.uniform(0.5, 2.0)
                if rng.random() < 0.5:
                    w = -w
                W[i, j] = w
    # Topological order
    remaining = set(range(d))
    topo_order = []
    while remaining:
        for node in sorted(remaining):
            parents_remaining = [p for p in range(d) if full_dag[p, node] == 1 and p in remaining]
            if not parents_remaining:
                topo_order.append(node)
                remaining.discard(node)
                break
    # Generate data
    X_all = np.zeros((2000, d), dtype=np.float64)
    for node in topo_order:
        parents = np.where(full_dag[:, node] == 1)[0]
        noise = rng.randn(2000)
        if len(parents) > 0:
            X_all[:, node] = X_all[:, parents] @ W[parents, node] + noise
        else:
            X_all[:, node] = noise
    # Binarize target
    median_val = np.median(X_all[:, target_idx])
    Y = (X_all[:, target_idx] > median_val).astype(np.float32)
    feature_cols = [i for i in range(d) if i != target_idx]
    X = X_all[:, feature_cols].astype(np.float32)

    return X, Y, config


def _load_insurance() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load INSURANCE-Gaussian dataset.

    Generates continuous data from the INSURANCE BN topology (27 nodes, 52 edges)
    using a Linear Gaussian SEM. Theft (binary node) is binarized as the
    classification target.
    """
    config = {**DATASET_CONFIGS["insurance"]}
    full_dag = _build_insurance_true_dag()
    config["true_dag"] = full_dag

    target_idx = config["target_idx"]
    rng = np.random.RandomState(43)  # Different seed from ALARM/CHILD
    d = full_dag.shape[0]
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if full_dag[i, j] == 1:
                w = rng.uniform(0.5, 2.0)
                if rng.random() < 0.5:
                    w = -w
                W[i, j] = w
    remaining = set(range(d))
    topo_order = []
    while remaining:
        for node in sorted(remaining):
            parents_remaining = [p for p in range(d) if full_dag[p, node] == 1 and p in remaining]
            if not parents_remaining:
                topo_order.append(node)
                remaining.discard(node)
                break
    X_all = np.zeros((2000, d), dtype=np.float64)
    for node in topo_order:
        parents = np.where(full_dag[:, node] == 1)[0]
        noise = rng.randn(2000)
        if len(parents) > 0:
            X_all[:, node] = X_all[:, parents] @ W[parents, node] + noise
        else:
            X_all[:, node] = noise
    median_val = np.median(X_all[:, target_idx])
    Y = (X_all[:, target_idx] > median_val).astype(np.float32)
    feature_cols = [i for i in range(d) if i != target_idx]
    X = X_all[:, feature_cols].astype(np.float32)

    return X, Y, config


# =============================================================================
# Public API
# =============================================================================

_LOADERS = {
    "lucas": _load_lucas,
    "sachs": _load_sachs,
    "asia": _load_asia,
    "diabetes": _load_diabetes,
    "heart_disease": _load_heart_disease,
    "breast_cancer": _load_breast_cancer,
    "child": _load_child,
    "neuropathic_pain": _load_neuropathic_pain,
    "alarm": _load_alarm,
    "insurance": _load_insurance,
}


def list_datasets() -> List[str]:
    """Return list of available dataset names."""
    return list(DATASET_CONFIGS.keys())


def get_dataset_info(name: str) -> Dict:
    """Get dataset configuration without loading data."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return {**DATASET_CONFIGS[name]}


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a benchmark dataset.

    Args:
        name: Dataset name (lucas, sachs, asia, diabetes, heart_disease, breast_cancer)

    Returns:
        X: Feature matrix (n_samples, n_features), float32
        Y: Target vector (n_samples,), float32
        config: Dict with metadata:
            - name: Human-readable name
            - feature_names: List of feature names
            - true_mb: Ground truth Markov Blanket indices (or None)
            - true_dag: Ground truth DAG matrix (or None)
            - has_true_dag: Whether ground truth is available
            - treatment_idx: Treatment variable index for DML (or None)
            - treatment_name: Treatment variable name (or None)
            - task: 'classification' or 'regression'
            - pop_size, n_gen, max_iter: Default NSGA-II config
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list_datasets()}")

    X, Y, config = _LOADERS[name]()

    # Validate
    assert X.ndim == 2, f"X should be 2D, got {X.ndim}D"
    assert Y.ndim == 1, f"Y should be 1D, got {Y.ndim}D"
    assert X.shape[0] == Y.shape[0], f"X/Y length mismatch: {X.shape[0]} vs {Y.shape[0]}"
    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"

    # Set defaults
    config.setdefault("true_dag", None)
    config.setdefault("true_mb", None)
    config.setdefault("has_true_dag", False)
    config.setdefault("treatment_idx", None)
    config.setdefault("treatment_name", None)
    config.setdefault("task", "classification")
    config.setdefault("skip_dml", False)

    # Auto-generate feature names if not set
    if config.get("feature_names") is None:
        config["feature_names"] = [f"X{i}" for i in range(X.shape[1])]

    # Ensure true_dag matches X dimensions (remove target node row/col)
    n_vars = X.shape[1]
    dag = config.get("true_dag")
    if dag is not None and dag.shape[0] > n_vars:
        target_idx = config.get("target_idx")
        if target_idx is not None:
            config["true_dag"] = np.delete(np.delete(dag, target_idx, axis=0), target_idx, axis=1)
        else:
            # Target assumed to be the last node (LUCAS convention)
            config["true_dag"] = dag[:n_vars, :n_vars]

    # Ensure feature_names matches X columns
    if len(config["feature_names"]) > n_vars:
        target_idx = config.get("target_idx")
        if target_idx is not None:
            config["feature_names"] = [
                n for i, n in enumerate(config["feature_names"]) if i != target_idx
            ]
        else:
            config["feature_names"] = config["feature_names"][:n_vars]

    return X, Y, config
