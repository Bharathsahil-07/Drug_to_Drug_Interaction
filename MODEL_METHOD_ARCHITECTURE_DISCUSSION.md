# Drug-Drug Interaction Prediction: Methods, Architecture, and Discussion

## Abstract
This project presents a hybrid Drug-Drug Interaction (DDI) prediction system that combines a known-interaction database with a Graph Attention Network (GAT) multi-task model. The pipeline first checks whether a drug pair exists in the curated interaction table. If the pair is already known, the system returns a database-backed high-risk result. If the pair is not found, the model predicts interaction probability, severity, and confidence from graph-structured drug representations. The architecture uses three stacked GAT layers and multi-head pair decoders for interaction, severity, and confidence outputs. In addition, post-hoc interpretability modules provide chemical similarity, ATC overlap, target/pathway overlap proxy, and graph-context evidence. The framework supports both terminal (CLI) and web/API workflows for practical DDI screening and analysis.

## Introduction
Drug-drug interaction prediction is a relational problem where each drug can influence the behavior of many others through shared targets, pathways, metabolism, and clinical context. Classical pairwise methods often miss neighborhood structure, while pure deep learning methods may ignore known clinical evidence if not designed carefully.

To address this, the project adopts a hybrid strategy:
1. Database-first safety layer: preserve known interactions as deterministic evidence.
2. Graph neural inference layer: predict unknown pairs using learned graph structure and node features.
3. Explainability layer: provide interpretable supporting signals for decision support.

This design is intended for practical use: it keeps known risks explicit while still enabling generalization to unseen combinations.

## All Methods Used
### 1. Data Parsing and Preparation
1. Drug and interaction records are parsed from DrugBank-derived files.
2. Core files used in runtime/training include:
- `data/drugs.csv` / `data/drugs_enriched.csv`
- `data/interactions.csv`
3. Optional enrichment adds SMILES and ATC codes when available.

### 2. Graph Construction
1. Drugs are nodes.
2. Known interactions are edges.
3. Node features combine:
- Categorical metadata (type/state/group)
- Text features from description/mechanism (TF-IDF)
- Chemical descriptors/fingerprints (when SMILES + RDKit available)
4. Edge attributes encode severity-like signal derived from interaction text.

### 3. Core Learning Method (GAT Multi-Task)
1. Encoder: Graph Attention Network (GATConv layers).
2. Decoder heads:
- Interaction head (binary link probability)
- Severity head (3-class output)
- Confidence head (0-1 score)
3. Training uses negative sampling for non-interaction examples.

### 4. Training and Evaluation Methods
1. Standard training pipeline in `train_v2.py`.
2. Rigorous protocols include:
- Cold-start (drug-level split; inductive setting)
- Warm-start (random edge split; transductive setting)
3. Metrics include AUC and classification-oriented measures in evaluation scripts.

### 5. Inference Method
In serving (`api_server.py`):
1. Check pair in known interactions database.
2. If not present, run model inference using graph embeddings.
3. Apply temperature scaling (calibration) to probability.
4. Map probability to risk categories.

### 6. Explainability Methods
1. Attention-based neighborhood influence (model-side).
2. Post-hoc feature contributions (`services/feature_importance.py`):
- Chemical similarity (Tanimoto)
- ATC similarity (prefix hierarchy)
- Target/pathway overlap proxy (token overlap)
- Graph context similarity (embedding cosine similarity)

## Architecture
### 1. System-Level Architecture
1. Data layer: drug table + interactions + graph artifacts.
2. Model layer: GAT encoder + multi-task decoders.
3. Service layer: Flask API endpoints for search/check/scan/graph.
4. Interface layer: Web dashboard + scan UI + CLI (`interaction_cli.py`).

### 2. Model Architecture (Technical)
From `mt_gat_model.py` and saved `trained_model_v2.pt` config:
1. GAT encoder:
- `conv1`: `GATConv(input_dim, hidden_dim, heads=4)`
- `conv2`: `GATConv(hidden_dim*heads, hidden_dim, heads=4)`
- `conv3`: `GATConv(hidden_dim*heads, embedding_dim, heads=1, concat=False)`
2. Typical v2 config:
- hidden_dim = 128
- embedding_dim = 64
- heads = 4
3. Pair decoding:
- Concatenate embeddings of drug A and drug B
- Predict interaction logit, severity logits, and confidence scalar

### 3. Runtime Decision Architecture
1. Deterministic branch: known pair -> database result.
2. Predictive branch: unknown pair -> model score + calibration + risk mapping.
3. Explanation branch: clinical text + graph attention context + feature breakdown.

## Model Discussions
### Strengths
1. Hybrid safety design: known evidence is preserved.
2. Graph-aware generalization for unknown drug pairs.
3. Multi-task output is clinically more informative than binary-only output.
4. Explainability support improves usability and trust.

### Limitations
1. Metadata availability affects explanation quality (SMILES/ATC/mechanism missingness).
2. Graph density can sometimes drive elevated probabilities without strong biochemical support.
3. Calibration currently uses fixed temperature and should be re-estimated per retrain.
4. Some product metrics in dashboards may be static placeholders and should be fully computed from artifacts.

### Practical Interpretation
1. Highest confidence: database-known interactions.
2. Strong model evidence: high probability with coherent feature overlap and neighborhood support.
3. Caution: moderate/high score but weak biological evidence should be treated as a hypothesis, not final clinical truth.

## What About Cold Start or Warm Start? What Is the Difference?
### Cold-Start (Inductive)
Implemented in `cold_start_trainer.py`.
1. Split strategy: by drugs (unseen drugs held out for test).
2. Training graph excludes held-out test-drug interactions from message passing.
3. Goal: test true generalization to new/unseen drugs.
4. Difficulty: harder setting, usually lower but more realistic performance.

### Warm-Start (Random Edge Split)
Implemented in `warm_start_rigorous_trainer.py`.
1. Split strategy: by edges (same drugs may appear in both train and test edges).
2. Training sees node representations for most/all test-time drugs.
3. Goal: evaluate link prediction when node identities are already known.
4. Difficulty: easier setting, often higher AUC due to node/structure familiarity.

### Key Difference Summary
1. Unit of split:
- Cold-start: split on drugs (nodes)
- Warm-start: split on interactions (edges)
2. Generalization target:
- Cold-start: unseen-drug generalization
- Warm-start: unseen-edge prediction among seen drugs
3. Leakage risk:
- Cold-start: reduced structural leakage
- Warm-start: higher chance of optimistic results
4. Recommended reporting:
- Report both; use cold-start as the stronger generalization benchmark.

## Conclusion
The project provides a practical and technically strong DDI framework: database-backed safety for known interactions, graph-based multi-task prediction for unknown interactions, and interpretable evidence layers for user-facing decision support. For rigorous model claims, both warm-start and cold-start results should be presented, with special emphasis on cold-start performance as the more realistic deployment proxy.
