"""Compatibility shim for canonical MT-GAT naming.

Primary implementation lives in gat_model.py.
This module provides stable imports:
- DrugInteractionMTGAT
- MTGATTrainer
"""

from gat_model import DrugInteractionMTGAT, MTGATTrainer
