# Drug Interaction Project - 26 Key Points (Detailed)

This document summarizes the project in 26 clear points with practical detail.

1. **Project Goal**
Build an AI-assisted system to identify potential drug-drug interactions and present risk in a clinically understandable way.

2. **Core Problem Solved**
Many drug pairs are difficult to evaluate quickly in real workflows. The system provides fast screening and explanation support.

3. **Hybrid Decision Strategy**
The pipeline first checks known interactions from the database, then uses the ML model for unknown pairs.

4. **Why Hybrid Is Important**
Known interactions should remain high-confidence references, while unknown pairs still receive model-based risk estimates.

5. **Model Type**
The serving model is a Graph Attention Network (GAT) multi-task model.

6. **Legacy Naming Clarification**
The class name is DrugInteractionMTGAT, but internal layers use GATConv (attention-based graph learning).

7. **Multi-Task Outputs**
The model predicts interaction probability, severity class, and confidence score from the same learned drug-pair representation.

8. **Why Graph Learning**
Drug interaction knowledge is relational. Graph methods use neighborhood context, not only isolated per-drug features.

9. **Node Definition**
Each drug is represented as a node in the graph.

10. **Edge Definition**
Each known drug-drug interaction is represented as an edge connecting two drug nodes.

11. **Node Feature Sources**
Features include metadata, text-derived features (description/mechanism), and chemical features when SMILES and RDKit are available.

12. **Edge Attribute Meaning**
Edge attributes include severity-like signals inferred from interaction descriptions.

13. **Training Pipeline**
Data parsing -> graph build -> model training -> checkpoint save -> API serving.

14. **Primary Model Artifacts**
The runtime typically uses data/trained_model_v2.pt with graph artifact data/drug_graph_v2.pt.

15. **Inference Entry Point**
The main runtime function is check_interaction_internal in the API server.

16. **Inference Flow**
Validate drug IDs -> database lookup -> model inference if needed -> risk mapping -> explanation assembly.

17. **Risk Mapping**
Risk labels are assigned by threshold mapping on calibrated probability (configuration-backed with fallback defaults).

18. **Probability Calibration**
Temperature scaling is used so reported probabilities are more realistic than raw logits.

19. **Clinical Explanation Layer**
The system produces mechanism-style clinical text based on metadata and interaction context.

20. **Model Explainability Layer**
Attention-based neighbor analysis identifies graph neighbors that most influenced a prediction.

21. **Feature Contribution Layer**
Chemical similarity, target overlap proxy, ATC similarity, and graph context similarity are provided when data is available.

22. **Dashboard Purpose**
The dashboard gives operational visibility: volume, risk distribution, recent activity, and model metric summaries.

23. **3D Graph Purpose**
The 3D graph helps users inspect network structure, hubs, and connectivity patterns behind graph-based predictions.

24. **Medicine Scan Pipeline**
Image preprocessing + OCR + fuzzy drug matching + interaction checks enables scan-to-risk workflows from medicine package images.

25. **Practical Interpretation Rule**
High model probability with weak biological evidence should be treated as cautionary (graph-driven hypothesis), not definitive mechanism proof.

26. **Clinical Usage Boundary**
This is a decision-support system. Final prescribing and safety actions must remain under professional clinical judgment.

---

## Suggested 60-Second Pitch
This project combines DrugBank data and graph attention AI to check drug interactions quickly, explain why a risk was predicted, and support safer decisions through API, dashboard, 3D graph exploration, and medicine-image scanning. It is designed for decision support, with calibration and explainability to make outputs more interpretable, but it does not replace clinician judgment.

