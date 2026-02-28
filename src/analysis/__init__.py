"""
audit_snd30.analysis
====================
Sous-package d'analyse statistique et sémantique.
"""

from .glissement import calculer_glissement
from .alignement import test_alignement

__all__ = ["calculer_glissement", "test_alignement"]
