# Model Deep Dive - Drug-Drug Interaction Predictor

Generated on: 2026-03-27
Scope: This document explains only the model stack in depth.

## 1. Model Purpose
The model predicts whether two drugs are likely to interact, and if so, with what severity pattern and confidence.

The implementation is in mt_gat_model.py using class DrugInteractionMTGAT.
Important note: the class name is legacy, but the encoder is Graph Attention Network based (GATConv layers).

## 2. Problem Formulation

### 2.1 Primary Task
Given two drugs A and B represented in a drug interaction graph, predict:
1. Interaction probability between A and B
2. Severity class distribution (Minor, Moderate, Major)
3. Confidence score

### 2.2 Why Graph Formulation
Drug interactions are relational:
1. Drugs are nodes.
2. Known interactions are edges.
3. Neighborhood context helps infer risk for unknown pairs.

A graph model can use both:
1. Local node features
2. Topological context from connected drugs

## 3. Input Representation

## 3.1 Graph Inputs
The model receives:
1. x: node feature matrix with shape [num_nodes, input_dim]
2. edge_index: message-passing edges [2, num_edges]
3. edge_label_index: target drug pairs to score [2, num_pairs]

## 3.2 Node Features (from graph_builder.py)
Node vectors combine multiple information sources:
1. Drug metadata signals:
- drug type indicators
- physical state indicators
- group indicators (approved/experimental/withdrawn)

2. Text-derived signals:
- TF-IDF features from drug description and mechanism text

3. Chemical signals (when available):
- molecular descriptors
- Morgan fingerprint bits (requires RDKit and valid SMILES)

## 3.3 Edge Attributes
Interaction descriptions are mapped to a severity-like scalar used as edge attribute during graph construction.
This provides additional supervision context for multi-task learning.

## 4. Encoder Architecture (GAT)

The encoder has three stacked GATConv blocks:

1. conv1 = GATConv(input_dim, hidden_dim, heads=heads)
2. conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
3. conv3 = GATConv(hidden_dim * heads, embedding_dim, heads=1, concat=False)

Activation and regularization:
1. ELU nonlinearity after conv1 and conv2
2. Dropout between attention layers

Output:
1. Node embeddings z with shape [num_nodes, embedding_dim]

Checkpoint-driven serving config (observed):
1. hidden_dim = 128
2. embedding_dim = 64
3. heads = 4

## 5. Pair Decoder And Multi-Task Heads

For each target pair (u, v):
1. Fetch embeddings z_u and z_v
2. Concatenate [z_u || z_v]
3. Feed to three heads

Heads:
1. Interaction head:
- MLP ending in one logit for binary interaction probability

2. Severity head:
- MLP ending in 3 logits for classes:
  - 0: Minor
  - 1: Moderate
  - 2: Major

3. Confidence head:
- MLP ending in sigmoid output in [0,1]

This enables richer prediction than single-task binary models.

## 6. Forward Pass Flow

Training/inference forward:
1. z = encode(x, edge_index)
2. (interaction_logits, severity_logits, confidence) = decode(z, edge_label_index)

For single-pair prediction:
1. Build edge_label_index with that pair only
2. Decode one pair and return probability + severity + confidence

## 7. Attention-Based Explainability

The encoder supports return_attention=True in encode.
When enabled, the final GAT layer returns:
1. edge index for attention output
2. attention weights

services/model_explainer.py uses this to compute:
1. Top influencing neighbors for each queried drug
2. Relative influence percentages
3. Heuristic confidence reasoning text

This is a key reason GAT was selected over non-attention baselines.

## 8. Training Objective (Conceptual)

The model is trained as multi-task learning over shared graph embeddings.

Typical components:
1. Binary interaction loss from interaction logits
2. Multi-class classification loss from severity logits
3. Regression loss from confidence output

The trainer in mt_gat_model.py handles edge splitting and optimization.
Negative sampling is used to create non-interaction examples for link prediction training.

## 9. Edge Splits And Label Construction

Trainer split strategy:
1. Unique undirected edges are extracted
2. Train/validation/test split applied on unique edges
3. Reverse edges added back for message passing

Severity labels are derived from edge attributes by threshold mapping to:
1. Minor
2. Moderate
3. Major

This provides supervision for the severity head.

## 10. Inference Path In API

Core runtime is in api_server.py function check_interaction_internal.

Decision path:
1. Check if pair exists in interactions.csv
- If yes: return database-backed result with source=database and high risk

2. If not in database:
- Map IDs to graph node indices
- Compute (or reuse cached) embeddings via model.encode
- Decode pair with model.decode
- Convert interaction logit to probability
- Map probability to risk band
- Add explainability outputs

This design balances deterministic known evidence and generalized model prediction.

## 11. Calibration Layer

The API applies temperature scaling:
1. model_temperature = 1.15
2. probability = sigmoid(interaction_logit / model_temperature)

Why this matters:
1. Raw neural probabilities can be overconfident.
2. Temperature scaling improves reliability of displayed probabilities.

UI label â€œCalibratedâ€ corresponds to this post-processing step.

## 12. Feature-Contribution Layer (Post-hoc)

services/feature_importance.py computes feature-space similarity scores:
1. Chemical similarity (Tanimoto on Morgan fingerprints)
2. ATC similarity (hierarchical prefix overlap)
3. Target overlap proxy (text-token overlap)
4. Graph context similarity (embedding cosine similarity)

These scores are not the modelâ€™s internal attention weights.
They are complementary post-hoc evidence to help interpretation.

## 13. Model Strengths

1. Graph-aware prediction for unseen pairs:
- Uses relational neighborhood context, not only direct pair metadata.

2. Multi-task output:
- Gives richer decision support than binary probability alone.

3. Explainability support:
- Attention neighbors + feature-contribution summaries help user trust.

4. Hybrid safety behavior:
- Known database interactions preserved as strongest signal.

## 14. Model Limitations

1. Data dependency:
- Missing SMILES/ATC/mechanism fields reduce interpretability quality.

2. Graph-driven false positives:
- Dense neighborhoods can produce moderate probabilities without strong biological support.

3. Calibration is fixed:
- Current temperature is static; should ideally be re-estimated per retrain/version.

4. Some API metric payloads are legacy:
- Certain reported metrics are static placeholders and should be replaced by computed artifacts.

## 15. Practical Interpretation Rules

For safe usage, interpret predictions with evidence context:

1. Highest confidence:
- Pair is known in database.

2. Strong model-supported case:
- High probability plus meaningful biological/feature support.

3. Caution case:
- Moderate/high probability but low biological evidence.
- Treat as hypothesis requiring monitoring/confirmation.

4. Low-risk case:
- Low probability and weak overlap evidence.

## 16. Improvement Roadmap For The Model Stack

1. Dynamic calibration pipeline:
- Learn temperature from validation set each retrain.

2. Reliability head or uncertainty estimation:
- Add explicit epistemic uncertainty for OOD/low-data pairs.

3. Stronger biological features:
- Better target/enzyme/transporter extraction from curated sources.

4. Better explainability consistency:
- Align post-hoc features with model attention evidence in a unified score.

5. Continuous evaluation reports:
- Publish computed metrics per model version and expose via API.

## 17. Quick Technical Glossary

1. GATConv:
- Graph attention convolution layer assigning learnable neighbor importance.

2. Link prediction:
- Predicting whether an edge should exist between two nodes.

3. Multi-task learning:
- One encoder shared by multiple prediction heads.

4. Temperature scaling:
- Post-hoc calibration technique for probability reliability.

5. Negative sampling:
- Creating likely non-edge pairs to train binary interaction discrimination.

---

## Final Summary
This model stack is a graph-attention multi-task DDI predictor with hybrid database fallback, calibrated probabilities, and layered explainability. It is strong for screening and decision support, especially when interpreted together with biological evidence and professional clinical judgment.

