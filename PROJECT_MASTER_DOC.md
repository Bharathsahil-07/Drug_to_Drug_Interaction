# Project Master Document

Generated on: 2026-03-27
Project root: Sdp/drug_interaction_project

## 1. Executive Summary
This project is an end-to-end Drug-Drug Interaction (DDI) analysis system that combines:
1. Graph-based machine learning (GAT multi-task model)
2. DrugBank-derived tabular and graph data
3. Clinical-style explanations and model explainability
4. Web API + dashboard + scan-from-image workflow

The system makes interaction decisions in two paths:
1. Database-first decision
- If a pair exists in interactions.csv, it is returned as a known interaction.
2. Model-based decision
- If a pair is not found in the database, the GAT model predicts probability and risk.

This hybrid design is used to maximize practical safety:
1. Known interactions should not be weakened by model uncertainty.
2. Unknown pairs still get probabilistic screening and explanations.

## 2. Method Used And Why

### 2.1 Core Method
The serving model is a Graph Attention Network (GAT) multi-task architecture implemented in mt_gat_model.py under class DrugInteractionMTGAT (legacy class name, GAT internals).

The model predicts:
1. Interaction likelihood (binary)
2. Severity class (Minor, Moderate, Major)
3. Confidence score (regression in 0 to 1)

### 2.2 Why GAT For DDI
DDI is naturally a graph problem:
1. Drugs are nodes.
2. Known interactions are edges.
3. Node neighborhood carries clinically relevant context.

GAT is suitable because:
1. It learns weighted neighbor influence (attention), not uniform averaging.
2. It supports explainability through influential-neighbor extraction.
3. It captures relational patterns beyond isolated feature vectors.

### 2.3 Why Multi-Task
Binary interaction alone is not enough for decision support.
Multi-task adds:
1. Severity for triage
2. Confidence for communication and filtering

## 3. Real End-To-End Workflow

### 3.1 Data Preparation Workflow
1. Parse DrugBank source into tabular files (drugs and interactions)
2. Optionally enrich metadata (SMILES, ATC, etc.)
3. Build graph with node features and edge attributes
4. Train multi-task GAT model
5. Save graph artifacts and trained checkpoints
6. Serve via Flask API and web UI

### 3.2 Training Workflow
1. Build graph in graph_builder.py
2. Node features include:
- Drug type/state/group indicators
- Text features from description and mechanism via TF-IDF
- Chemical features if SMILES and RDKit are available
3. Edge severity attribute is inferred from interaction description text
4. Train in train_v2.py
5. Save data/trained_model_v2.pt and data/drug_graph_v2.pt

### 3.3 Runtime Inference Workflow
Core function: check_interaction_internal in api_server.py

Step-by-step:
1. Validate both drug IDs are present
2. Search known pair in interactions database
- If found, return database-based interaction result
3. If not found:
- Map drug IDs to graph indices
- Use cached embeddings from model.encode
- Decode pair with model.decode
- Apply temperature calibration
- Convert probability to risk label
- Add clinical explanation + model explanation + feature contributions

## 4. How The Model Predicts And Why

### 4.1 Graph Encoding
The model uses three GATConv layers:
1. Layer 1: input to hidden with multi-head attention
2. Layer 2: hidden to hidden with multi-head attention
3. Layer 3: hidden to embedding with single head output

Each drug gets an embedding that summarizes:
1. Its own feature profile
2. Interaction-context from neighboring drugs

### 4.2 Pair Scoring
For drug pair A and B:
1. Take embedding of A and B
2. Concatenate embeddings
3. Pass through three heads:
- Interaction head: binary logit
- Severity head: 3-class logits
- Confidence head: scalar confidence

### 4.3 Calibration
API applies temperature scaling with model_temperature = 1.15.
Displayed probability is calculated from:
1. interaction_logit divided by temperature
2. sigmoid transform to probability

Purpose:
1. Improve probability reliability
2. Reduce overconfidence in raw neural outputs

### 4.4 Why A Pair May Be Predicted As Interacting
A pair can receive non-trivial probability because:
1. Graph topology places both drugs near interaction-dense neighborhoods
2. Shared latent patterns exist in embedding space
3. Feature overlap contributes (when metadata is available)

When biological features are weak but graph signal is strong, prediction can still be moderate.
Such cases should be interpreted as graph-driven hypotheses requiring monitoring and validation.

## 5. Dashboard Explanation

Dashboard file: static/dashboard.html
Main endpoint: /api/dashboard/stats

### 5.1 What The Dashboard Shows
1. Overview cards
- Total drugs
- Total interactions
- Model AUC display
- Searches today

2. Charts
- Interaction distribution
- Risk-level distribution

3. Model metrics table
- AUC, sensitivity, specificity, parameters, and related fields from endpoint

4. Top dangerous combinations
- Example combinations generated for dashboard visibility

5. Real-time activity panel
- Recent search and interaction-check activity from in-memory tracking

### 5.2 Why Dashboard Exists
1. Provides operational visibility for users and developers
2. Shows system usage trends and interaction-check workload
3. Supports model transparency at product level

## 6. 3D Graph Explanation

Endpoint: /api/graph/3d

### 6.1 What It Does
1. Reads graph_data from loaded model artifacts
2. Selects top nodes by degree (limit parameter)
3. Builds a filtered subgraph
4. Finds connected components
5. Returns node/link payload for interactive 3D visualization

### 6.2 Why It Is Useful
1. Makes graph structure visible
2. Highlights hubs and clusters influencing model behavior
3. Helps explain graph-driven predictions to users

### 6.3 Offline Visualization
visualize.py can generate:
1. Interactive HTML network
2. Static PNG graphs
3. Community and PageRank-based visual emphasis

## 7. Scan Medicine Workflow (Image To Interaction)

Blueprint: routes/scan_routes.py
Services used:
1. services/image_preprocessing.py
2. services/ocr_service.py
3. services/drug_match_service.py
4. check_interaction_internal through injected dependency

### 7.1 Detailed Flow
1. User uploads one or many medicine images
2. Images are preprocessed:
- grayscale conversion
- resizing
- contrast/brightness tuning
- sharpening
- adaptive thresholding
3. OCR extraction with EasyOCR:
- GPU enabled when available
- fallback to inverted-image OCR if text is missed
4. Drug matching pipeline:
- OCR cleanup and line filtering
- blacklist-based noise rejection
- exact and fuzzy name matching
- alias map for brand-name variants and OCR errors
- global fallback for common generic names
5. Interaction generation:
- If multiple detected drugs: pairwise interaction checks
- If one detected drug: suggestion mode against common drugs
6. Response includes:
- detected drug list
- confidence details
- interaction objects
- graph nodes/links for UI rendering

### 7.2 Why This Design
1. OCR on packaging is noisy and inconsistent
2. Multi-stage matching improves recall and precision
3. Pairwise and fallback modes support real user scenarios

## 8. Explainability Layers

### 8.1 Clinical Explanation (services/interaction_explainer.py)
Provides:
1. Mechanism-style text from interaction descriptions and metadata
2. Interaction type labeling (pharmacokinetic/pharmacodynamic/combined)
3. Human-readable clinical reasoning and disclaimer
4. Consistency checks to reduce contradictory narratives

### 8.2 Model Explanation (services/model_explainer.py)
Provides:
1. Top influencing graph neighbors
2. Relative influence percentages from attention weights
3. Confidence reasoning text based on attention concentration

### 8.3 Feature Contributions (services/feature_importance.py)
Computes:
1. Chemical similarity (RDKit Tanimoto if SMILES available)
2. Target overlap proxy from mechanism text
3. ATC similarity from hierarchical prefix matching
4. Graph context similarity from embedding cosine similarity

## 9. Data Assets And Their Roles

1. data/drugs.csv
- Core parsed drug metadata

2. data/drugs_enriched.csv
- Extended metadata with smiles and atc_code columns (availability depends on enrichment quality)

3. data/interactions.csv
- Known DDI database used for direct lookup and graph edges

4. data/drug_graph.pt and data/drug_graph_v2.pt
- Serialized graph objects and mappings

5. data/trained_model.pt and data/trained_model_v2.pt
- Saved model checkpoints (v2 preferred when present)

6. data/drugbank_smiles_atc_all.csv
- Full extraction from DrugBank XML (SMILES/InChI/InChIKey/ATC where available)

7. data/drugbank_smiles_atc_ml.csv
- ML subset where both SMILES and ATC are present

## 10. API Surfaces In Product Workflow

Most important endpoints:
1. GET /api/health
- Liveness and load status

2. GET /api/drugs/search
- Search drugs by query text

3. POST /api/interactions/check
- Main DDI endpoint for interactive use

4. POST /api/interactions/batch
- Batch interaction checks

5. GET /api/dashboard/stats
- Dashboard payload

6. GET /api/graph/3d
- 3D graph data

7. POST /api/scan-drug-image
- Image-based medicine recognition and interaction pipeline

## 11. Python Files In Project And Their Role In Workflow

### 11.1 Core Serving And Product
1. api_server.py
- Main Flask app, model loading, routing, interaction core logic
2. routes/scan_routes.py
- Scan image API blueprint and response assembly
3. interaction_cli.py
- CLI access to search and interaction features

### 11.2 Model And Training
1. mt_gat_model.py
- GAT multi-task architecture and trainer components
2. train_v2.py
- Current training pipeline and model checkpoint creation
3. cold_start_trainer.py
- Cold-start training/evaluation routines
4. warm_start_rigorous_trainer.py
- Warm-start variant for rigorous runs
5. scaffold_rigorous_trainer.py
- Scaffolding helper for rigorous trainer flow
6. ablation_study_manager.py
- Ablation study execution and comparisons
7. comprehensive_evaluation.py
- Broader metric and calibration evaluations
8. evaluate_model.py
- Standard model evaluation script
9. rigorous_final_evaluator.py
- Strict final evaluation script

### 11.3 Data And Graph Construction
1. data_parser.py
- DrugBank parsing to tabular format
2. graph_builder.py
- Build graph edges, node features, and PyG graph objects
3. rigorous_graph_builder.py
- Rigorous/alternate graph build pipeline
4. data_enrichment.py
- Metadata enrichment helper (for example via PubChem)
5. extract_smiles_atc_from_xml.py
- Extraction of SMILES/InChI/InChIKey/ATC from XML
6. check_xml_smiles_atc.py
- Targeted XML check utility
7. fast_xml_scan.py
- Fast XML scanner utility
8. sample_xml_data.py
- XML sampling utility for completeness checks

### 11.4 Explainability And Clinical Reasoning
1. services/interaction_explainer.py
- Clinical interpretation layer and consistency filtering
2. services/model_explainer.py
- Attention-neighbor influence extraction
3. services/feature_importance.py
- Feature-contribution and similarity calculations

### 11.5 OCR And Drug Matching
1. services/image_preprocessing.py
- OCR-oriented image enhancements
2. services/ocr_service.py
- Text extraction with EasyOCR and fallback
3. services/drug_match_service.py
- Multi-stage matching and confidence scoring

### 11.6 Visualization, Analysis, Reporting
1. visualize.py
- Network visualizations (interactive/static)
2. latent_space_visualizer.py
- Embedding-space visualization
3. interpretability_analyzer.py
- Additional interpretation analyses
4. diversity_analysis.py
- Diversity and robustness analysis
5. publication_plotter.py
- Publication-oriented plotting
6. publication_final_generator.py
- Publication artifact generation
7. get_metrics.py
- Metrics extraction and display
8. show_metrics_info.py
- Human-readable metric explanation
9. log_converter.py
- Log conversion utility
10. package_model.py
- Model artifact packaging utility

### 11.7 Orchestration And Demo Scripts
1. run_pipeline.py
- End-to-end parse-build-train pipeline runner
2. demo_features.py
- Feature demonstration script
3. FEATURE_SHOWCASE.py
- API feature showcase script
4. find_drugs.py
- Drug lookup helper
5. inspect_model.py
- Model checkpoint and architecture inspection

### 11.8 Validation And Test Scripts
1. test_api.py
- API tests
2. test_all_features.py
- End-to-end feature tests
3. test_graph_endpoint.py
- Graph endpoint tests
4. test_graph_endpoint_large.py
- Large graph endpoint tests

### 11.9 Local Utilities Added During Debugging
1. data/check_csv_columns.py
- CSV schema inspection helper
2. data/analyze_smiles_atc.py
- SMILES and ATC completeness analysis
3. build_master_doc.py
- Legacy doc builder utility (current document is manually curated for accuracy)

## 12. Interpretation Guidance For Real Use

### 12.1 Clinically Stronger Cases
1. Pair exists in known interaction database
2. Model probability high and biological features support mechanism

### 12.2 Caution Cases
1. Moderate/high graph probability but weak biological overlap
2. Missing structural/classification metadata reducing feature support

### 12.3 Recommended Safety Messaging
1. Use as decision-support, not autonomous prescribing
2. Monitor uncertain cases
3. Confirm with professional clinical judgment and references

## 13. Known Limitations
1. Some dashboard and metrics fields are static or heuristic in current implementation.
2. Feature decomposition quality depends on metadata completeness (SMILES, ATC, mechanism text).
3. Clinical explanation text is rule-based and should not be treated as a formal guideline.

## 14. Recommended Next Improvements
1. Replace static metric payload fields with computed values from evaluation artifacts.
2. Add explicit reliability score combining graph evidence and biological evidence availability.
3. Add per-feature availability diagnostics in every response path for UI transparency.
4. Maintain versioned API schema documentation for frontend and external clients.

## 15. Final One-Paragraph Project Story
This project is a hybrid DDI intelligence platform that first uses known DrugBank interactions for deterministic safety coverage, then applies a graph-attention multi-task model for unseen pairs, and enriches outputs with clinical reasoning, feature-based explanation, dashboard analytics, 3D graph exploration, and OCR-powered medicine scanning so users can move from raw data to practical, explainable interaction decisions.

