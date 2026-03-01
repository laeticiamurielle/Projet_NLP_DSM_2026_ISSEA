"""
audit_snd30.nlp.embeddings
===========================
Utilitaires centralisés pour les embeddings de phrases
via SentenceTransformer.

Ce module permet de charger une seule fois le modèle
configuré dans audit_snd30.config et de produire des
vecteurs numpy à partir de listes de textes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from audit_snd30.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Charge et met en cache le modèle SentenceTransformer.

    Le nom du modèle est défini dans EMBEDDING_MODEL du config.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Retourne les embeddings d'une séquence de textes.

    Paramètres
    ----------
    texts      : séquence de chaînes de caractères
    batch_size : taille de batch pour l'encodage (performance)

    Retour
    ------
    np.ndarray de forme (n_textes, dim_embedding)
    """

    model = get_embedding_model()
    # SentenceTransformer renvoie déjà un array numpy
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings
