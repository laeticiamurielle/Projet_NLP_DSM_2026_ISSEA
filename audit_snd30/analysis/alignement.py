"""
audit_snd30.analysis.alignement
================================
ÉTAPE 4 — Test d'alignement statistique du budget avec la SND30
via le test du Chi² de Pearson.

Hypothèses
----------
H₀ : la distribution observée = cibles SND30 (budget ALIGNÉ)
H₁ : écart significatif        (budget DÉSALIGNÉ)

Règle de décision
-----------------
p-value > 0.05 → on conserve H₀ → ✓ Aligné
p-value < 0.05 → on rejette H₀  → ✗ Désaligné
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from audit_snd30.config import PILIERS, CIBLES_SND30


def test_alignement(
    df_2024: pd.DataFrame,
    df_2025: pd.DataFrame,
) -> pd.DataFrame:
    """
    Teste si la distribution budgétaire observée est alignée
    avec les cibles théoriques de la SND30.

    Le test compare le nombre de lignes observées dans chaque pilier
    avec les effectifs attendus calculés à partir des proportions SND30.

    Paramètres
    ----------
    df_2024 / df_2025 : DataFrames classifiés (colonne PILIER_SND30 requise)

    Retour
    ------
    DataFrame avec une ligne par année :
      ANNEE, CHI2, P_VALUE, ALIGNEMENT, INTERPRETATION, ECARTS_PAR_PILIER
    """
    lignes = []

    for annee, df in [(2024, df_2024), (2025, df_2025)]:
        # Distribution observée (effectifs par pilier)
        observed = np.array(
            [len(df[df["PILIER_SND30"] == p]) for p in PILIERS], dtype=float
        )
        total    = observed.sum()

        # Effectifs attendus selon les cibles SND30
        expected = np.array([CIBLES_SND30[p] * total for p in PILIERS])

        # Test Chi²
        chi2, p_value = chisquare(f_obs=observed, f_exp=expected)

        # Écart en points de % entre observé et cible
        ecarts = {
            p: round((observed[i] / total * 100) - (CIBLES_SND30[p] * 100), 2)
            for i, p in enumerate(PILIERS)
        }

        aligne = p_value > 0.05
        lignes.append({
            "ANNEE":             annee,
            "CHI2":              round(float(chi2), 4),
            "P_VALUE":           round(float(p_value), 4),
            "ALIGNEMENT":        "✓ Aligné" if aligne else "✗ Désaligné",
            "INTERPRETATION":    (
                f"LF {annee} : alignement non significatif (p={p_value:.3f} > 0.05)"
                if aligne else
                f"LF {annee} : désalignement statistiquement prouvé (p={p_value:.3f} < 0.05)"
            ),
            "ECARTS_PAR_PILIER": ecarts,
        })

    return pd.DataFrame(lignes)
