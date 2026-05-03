# Drug-Drug Interaction Prediction with Multi-Task Graph Attention Networks

## Abstract
Polypharmacy increases the risk of harmful drug-drug interactions (DDIs), creating a need for computational screening tools that can prioritize risky combinations. This project implements a hybrid DDI framework that combines a database-first safety layer with a multi-task Graph Attention Network (MT-GAT). In the serving pipeline, known interactions from the interaction database are returned directly, while unknown pairs are scored by the model. The MT-GAT encoder uses three stacked graph-attention layers to learn drug embeddings from graph structure and node features, and jointly predicts interaction probability, severity class, and confidence. Feature engineering includes pharmacological metadata, text-derived representations, and optional chemical descriptors/fingerprints when SMILES are available. The project also includes rigorous cold-start and warm-start training/evaluation scripts, along with explainability modules (attention-neighbor analysis and post-hoc feature contributions). This document reports only methods and artifacts that are implemented in the repository; numerical results should be filled from successful reruns of the corresponding scripts.

**Index Terms**: drug-drug interaction, graph attention network, multi-task learning, inductive evaluation, cold-start, warm-start, pharmacovigilance.

---

## I. Introduction
Drug-drug interactions can cause severe adverse events, especially in patients taking multiple medications. The repository addresses this with a hybrid design:
1. Database-first path for known DDIs.
2. Model-based path for unknown DDIs.

The model path is implemented with a multi-task GAT architecture to predict interaction risk and severity while preserving interpretability through attention and feature-based explanations.

---

## II. Related Work
### A. Graph Neural Networks for DDI
The project is aligned with graph-based DDI prediction literature and implements a GAT-style architecture for pair scoring over a drug interaction graph.

### B. Inductive vs. Transductive Evaluation
This repository explicitly includes both settings:
1. Cold-start (inductive, unseen drugs): `cold_start_trainer.py`.
2. Warm-start (random edge split among seen drugs): `warm_start_rigorous_trainer.py`.

### C. Multi-Task Learning
The model jointly predicts interaction, severity, and confidence, using shared graph embeddings and task-specific heads.

---

## III. Methodology
### A. Data and Node Feature Construction
Implemented sources and processing:
1. Drug records: `data/drugs.csv` and optionally `data/drugs_enriched.csv`.
2. Interaction records: `data/interactions.csv`.
3. Node features in `graph_builder.py` / `rigorous_graph_builder.py` include:
- Pharmacological metadata indicators.
- Text features (TF-IDF).
- Optional chemical features (RDKit descriptors + Morgan fingerprint) when SMILES are present.

### B. Leakage-Controlled Rigorous Pipeline
In `rigorous_graph_builder.py`, `StandardScaler` and `TfidfVectorizer` are fitted on train-drug indices only, then applied to all nodes. This supports leakage-reduced inductive evaluation for the rigorous scripts.

### C. Graph Construction and Severity Encoding
Graph definition:
1. Nodes: drugs.
2. Edges: known interactions (stored in both directions).
3. Edge severity score from interaction-description keyword rules:
- Critical-like keywords -> 1.0
- Major-like keywords -> 0.7
- Minor/moderate/caution-like keywords -> 0.4
- Default -> 0.5

Severity class weighting for imbalance is implemented in `rigorous_graph_builder.py` via inverse-frequency weights.

### D. MT-GAT Architecture
Implemented in `gat_model.py` (exposed via `mt_gat_model.py`):
1. Encoder:
- GATConv Layer 1: hidden_dim with multi-head attention.
- GATConv Layer 2: hidden_dim with multi-head attention.
- GATConv Layer 3: embedding_dim, single head (concat=False).
2. Pair embedding: concatenation of source and target node embeddings.
3. Heads:
- Interaction head (binary logit)
- Severity head (3-class logits)
- Confidence head (sigmoid regression output)

### E. Training Objective
Composite loss implemented in `gat_model.py`:

$$
L = L_{interaction} + 0.5 \cdot L_{severity} + 0.2 \cdot L_{confidence}
$$

Where:
1. Interaction loss uses BCE with logits on positive + sampled negative edges.
2. Severity loss uses cross-entropy on positive edges.
3. Confidence loss uses MSE toward binary edge labels.

### F. Optimization and Model Selection
Default setup in project scripts:
1. Optimizer: Adam.
2. Learning rate: 0.001.
3. Weight decay: 5e-4.
4. Early stopping is implemented in `MTGATTrainer.train()`.

---

## IV. Experimental Results (Project-Aligned Reporting)
### A. Current Status
The repository contains scripts for ablation, diversity, calibration, and rigorous evaluation; however, several result text logs currently show environment/runtime failures (memory/DLL issues). Therefore, values below should be filled only from successful reruns.

### B. Current Real Metrics From Model Run
The following MT-GAT values are taken from `rigorous_metrics_all.txt` generated by the rigorous evaluator. Comparison columns remain N/A until a complete paired single-task run is finalized in the current environment.

| Metric | MT-GAT | Delta | p-value |
|---|---:|---:|---:|
| AUC-ROC | 0.9390 +- 0.0128 | N/A | N/A |
| AUPRC | 0.9329 +- 0.0119 | N/A | N/A |
| F1 | 0.2497 +- 0.0931 | N/A | N/A |
| Precision | N/A | N/A | N/A |
| Recall | N/A | N/A | N/A |
| Brier | 0.4144 +- 0.0296 | N/A | N/A |
| ECE | 0.4263 +- 0.0278 | N/A | N/A |
| Severity Macro-F1 | 0.3676 +- 0.1018 | N/A | N/A |

If needed for completeness, you can also cite:
1. Accuracy: 0.5716 +- 0.0307
2. Severity Weighted-F1: 0.7504 +- 0.1316



---

## V. Discussion
### A. Strengths of Current Implementation
1. Hybrid safety-first serving behavior (known DDIs preserved).
2. Multi-task outputs provide richer decision support.
3. Cold-start and warm-start scripts are both present for robust evaluation.
4. Explainability stack is implemented (attention and feature-based evidence).

### B. Current Limitations
1. Some published-style result logs in repo are from failed runs and must not be treated as final.
2. Calibration in serving is currently fixed-temperature in API path.
3. Full benchmark reproducibility requires stable environment reruns.

### C. Practical Next Step
Rerun ablation/diversity/calibration scripts in a stable environment, then replace placeholders with verified values and confidence intervals.

---

## VI. Figure Placement Plan (Exact Paths)
Use these figures in this paper order.

| Figure No. | Caption Suggestion | File Path |
|---|---|---|
| Fig. 1 | Model Architecture Overview | `publication_figures/1_architecture.png` |
| Fig. 2 | ROC Curves (Model Comparison) | `publication_figures/2_roc_curves.png` |
| Fig. 3 | Precision-Recall Curves | `publication_figures/3_pr_curves.png` |
| Fig. 4 | Calibration Plot | `publication_figures/4_calibration.png` |
| Fig. 5 | Similarity-Based Generalization | `publication_figures/5_similarity_generalization.png` |
| Fig. 6 | Ablation Metrics Summary | `publication_figures/6_ablation_metrics.png` |
| Fig. 7 | Confusion Matrix | `publication_figures/7_confusion_matrix.png` |

Backup/alternate figure sources already present in project:
1. `plots/roc_curves.png`
2. `plots/pr_curves.png`
3. `plots/calibration.png`
4. `plots/diversity_analysis.png`
5. `plots/db00363_attention_hub.png`
6. `plots_rigorous/roc_rigorous.png`
7. `plots_rigorous/pr_rigorous.png`
8. `plots_rigorous/calibration_rigorous.png`
9. `plots_rigorous/cm_raw_rigorous.png`

---

## VII. Files Mapped to Methods (Traceability)
1. Model architecture and losses: `gat_model.py`, `mt_gat_model.py`
2. Standard graph/features: `graph_builder.py`
3. Leakage-reduced rigorous graph/features: `rigorous_graph_builder.py`
4. Main training pipeline: `train_v2.py`
5. Cold-start inductive training: `cold_start_trainer.py`
6. Warm-start random-edge training: `warm_start_rigorous_trainer.py`
7. Ablation script and stats test: `ablation_study_manager.py`
8. Serving + calibration path: `api_server.py`
9. Explainability services: `services/model_explainer.py`, `services/feature_importance.py`, `services/interaction_explainer.py`

---

## VIII. How to Use This Draft
1. Keep sections I-VII as the master structure.
2. Fill metric placeholders only from successful script runs.
3. Keep figure order from Section VI for consistent narrative flow.
4. Do not claim numbers that are only present in failed logs.
