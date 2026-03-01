"""
audit_snd30.analysis.embeddings_explorer
========================================
Outils d'exploration visuelle des embeddings SentenceTransformer.

Fonctions principales
---------------------
    - construire_embeddings_df :
        à partir d'un DataFrame classifié, calcule les embeddings,
        applique UMAP (2D) et renvoie un DataFrame prêt à être
        sauvegardé ou visualisé (x, y, pilier, année, libellé).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from umap import UMAP

from audit_snd30.config import PILIERS
from audit_snd30.nlp.embeddings import embed_texts


def _sample_by_pilier(
    df: pd.DataFrame,
    label_col: str,
    max_par_pilier: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Échantillonne au plus `max_par_pilier` lignes par pilier.

    Si max_par_pilier est None, renvoie le DataFrame complet.
    """

    if max_par_pilier is None:
        return df
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_par_pilier), random_state=random_state))
    )


def construire_embeddings_df(
    df: pd.DataFrame,
    text_col: str = "LIBELLE",
    label_col: str = "PILIER_SND30",
    year_col: str = "ANNEE",
    max_par_pilier: Optional[int] = 500,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 42,
) -> pd.DataFrame:
    """Construit un DataFrame 2D d'exploration des embeddings.

    Étapes
    ------
    1. Échantillonnage équilibré par pilier (optionnel)
    2. Encodage SentenceTransformer → vecteurs haute dimension
    3. Projection UMAP en 2D

    Retour
    ------
    DataFrame avec colonnes :
        - x, y        : coordonnées UMAP
        - PILIER_SND30
        - ANNEE (si présente)
        - LIBELLE
    """

    # Filtrer uniquement les lignes dont le pilier est connu
    df_work = df[df[label_col].isin(PILIERS)].copy()
    df_work = _sample_by_pilier(df_work, label_col, max_par_pilier)

    textes = df_work[text_col].fillna("").astype(str).tolist()
    embeddings = embed_texts(textes)

    reducer = UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=umap_random_state,
    )
    coords = reducer.fit_transform(embeddings)

    df_emb = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            label_col: df_work[label_col].tolist(),
            text_col: df_work[text_col].tolist(),
        }
    )

    if year_col in df_work.columns:
        df_emb[year_col] = df_work[year_col].tolist()

    return df_emb
