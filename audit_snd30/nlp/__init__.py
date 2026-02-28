"""
audit_snd30.nlp
===============
Sous-package de classification NLP (CamemBERT zero-shot + fine-tuning).
"""

from .classification import zero_shot, fine_tuner, predire

__all__ = ["zero_shot", "fine_tuner", "predire"]
