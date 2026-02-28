"""
audit_snd30.analysis.glissement
================================
ÉTAPE 3 — Mesurer mathématiquement l'évolution des priorités
budgétaires 2024 → 2025 via trois métriques NLP complémentaires.

Métriques
---------
1. Jensen-Shannon   — distance entre distributions de piliers
2. TF-IDF Cosinus   — évolution du vocabulaire par pilier
3. Δ Part AE/CP     — transfert de ressources financières
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PILIERS


def _distribution(df: pd.DataFrame) -> np.ndarray:
    """Proportion de lignes dans chaque pilier (vecteur de probabilité)."""
    counts = df["PILIER_SND30"].value_counts(normalize=True)
    return np.array([counts.get(p, 0.0) for p in PILIERS])


def _parts_budget(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Part (%) de chaque pilier dans un montant budgétaire total."""
    total = df[col].sum()
    if total == 0:
        return {p: 0.0 for p in PILIERS}
    return {
        p: round(df[df["PILIER_SND30"] == p][col].sum() / total * 100, 2)
        for p in PILIERS
    }


def calculer_glissement(
    df_2024: pd.DataFrame,
    df_2025: pd.DataFrame,
    ae_col: str = "AE",
    cp_col: str = "CP",
) -> dict:
    """
    Calcule le glissement sémantique et budgétaire entre LF 2024 et LF 2025.

    Paramètres
    ----------
    df_2024 / df_2025 : DataFrames classifiés (colonnes PILIER_SND30, AE, CP)
    ae_col / cp_col   : noms des colonnes montants

    Retour
    ------
    dict structuré avec toutes les métriques :
      - distributions       : part lignes par pilier par année
      - jensen_shannon      : score JS [0, 1]
      - jensen_shannon_interpretation : texte lisible
      - cosine_tfidf        : similarité vocab par pilier
      - parts_ae / parts_cp : répartition budgétaire (%)
      - delta_ae / delta_cp : variation en points de %
      - montants_ae / montants_cp : montants absolus
    """
    resultats: dict = {}

    # ── Distribution des piliers ──────────────────────────────────────────────
    dist_2024 = _distribution(df_2024)
    dist_2025 = _distribution(df_2025)
    resultats["distributions"] = {
        2024: dict(zip(PILIERS, dist_2024.round(4).tolist())),
        2025: dict(zip(PILIERS, dist_2025.round(4).tolist())),
    }

    # ── Jensen-Shannon ────────────────────────────────────────────────────────
    # Formule : JS(P||Q) = 0.5·KL(P||M) + 0.5·KL(Q||M), M = (P+Q)/2
    # Interprétation : 0 = distributions identiques, 1 = totalement différentes
    p = np.clip(dist_2024, 1e-10, None); p /= p.sum()
    q = np.clip(dist_2025, 1e-10, None); q /= q.sum()
    js_score = float(jensenshannon(p, q))
    resultats["jensen_shannon"] = round(js_score, 4)
    resultats["jensen_shannon_interpretation"] = (
        "Faible glissement (priorités stables)"          if js_score < 0.10 else
        "Glissement modéré (réorientation partielle)"    if js_score < 0.25 else
        "Fort glissement (rupture de priorités)"
    )

    # ── Similarité cosinus TF-IDF par pilier ─────────────────────────────────
    # Pour chaque pilier : compare le corpus de libellés 2024 vs 2025
    cosine_par_pilier: dict[str, float | None] = {}
    for pilier in PILIERS:
        t24 = df_2024[df_2024["PILIER_SND30"] == pilier]["LIBELLE"].tolist()
        t25 = df_2025[df_2025["PILIER_SND30"] == pilier]["LIBELLE"].tolist()
        if not t24 or not t25:
            cosine_par_pilier[pilier] = None
            continue
        vect = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        mat  = vect.fit_transform([" ".join(t24), " ".join(t25)])
        cosine_par_pilier[pilier] = round(float(cosine_similarity(mat[0], mat[1])[0][0]), 4)
    resultats["cosine_tfidf"] = cosine_par_pilier

    # ── Parts budgétaires et delta ────────────────────────────────────────────
    parts_ae_2024 = _parts_budget(df_2024, ae_col)
    parts_ae_2025 = _parts_budget(df_2025, ae_col)
    parts_cp_2024 = _parts_budget(df_2024, cp_col)
    parts_cp_2025 = _parts_budget(df_2025, cp_col)

    resultats["parts_ae"] = {2024: parts_ae_2024, 2025: parts_ae_2025}
    resultats["parts_cp"] = {2024: parts_cp_2024, 2025: parts_cp_2025}
    resultats["delta_ae"] = {p: round(parts_ae_2025[p] - parts_ae_2024[p], 2) for p in PILIERS}
    resultats["delta_cp"] = {p: round(parts_cp_2025[p] - parts_cp_2024[p], 2) for p in PILIERS}

    # ── Montants absolus ──────────────────────────────────────────────────────
    resultats["montants_ae"] = {
        yr: {p: int(df[df["PILIER_SND30"] == p][ae_col].sum()) for p in PILIERS}
        for yr, df in [(2024, df_2024), (2025, df_2025)]
    }
    resultats["montants_cp"] = {
        yr: {p: int(df[df["PILIER_SND30"] == p][cp_col].sum()) for p in PILIERS}
        for yr, df in [(2024, df_2024), (2025, df_2025)]
    }

    return resultats
