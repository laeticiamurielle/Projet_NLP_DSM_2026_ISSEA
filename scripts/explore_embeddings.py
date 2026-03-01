"""scripts/explore_embeddings.py
================================
Génère un fichier CSV d'exploration des embeddings SentenceTransformer
à partir des DataFrames classifiés produits par la pipeline.

Usage
-----
    poetry run python scripts/explore_embeddings.py \
        --df-2024 data/processed/df_2024_classifie.xlsx \
        --df-2025 data/processed/df_2025_classifie.xlsx \
        --out data/processed/embeddings_umap.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audit_snd30.config import PROC_DIR
from audit_snd30.analysis.embeddings_explorer import construire_embeddings_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploration UMAP des embeddings SND30")
    parser.add_argument(
        "--df-2024",
        type=Path,
        default=PROC_DIR / "df_2024_classifie.xlsx",
        help="Chemin vers le fichier classifié 2024 (Excel)",
    )
    parser.add_argument(
        "--df-2025",
        type=Path,
        default=PROC_DIR / "df_2025_classifie.xlsx",
        help="Chemin vers le fichier classifié 2025 (Excel)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC_DIR / "embeddings_umap.csv",
        help="Chemin de sortie du CSV (embedddings UMAP)",
    )
    parser.add_argument(
        "--max-par-pilier",
        type=int,
        default=500,
        help="Nombre maximum de lignes par pilier et par année pour l'UMAP",
    )

    args = parser.parse_args()

    if not args.df_2024.exists() or not args.df_2025.exists():
        raise FileNotFoundError(
            "Fichiers classifiés introuvables. Lance d'abord scripts/run_pipeline.py"
        )

    df_2024 = pd.read_excel(args.df_2024)
    df_2025 = pd.read_excel(args.df_2025)

    df_all = pd.concat([df_2024, df_2025], ignore_index=True)

    print("[explore_embeddings] Calcul des embeddings et de l'UMAP 2D...")
    df_emb = construire_embeddings_df(
        df_all,
        text_col="LIBELLE",
        label_col="PILIER_SND30",
        year_col="ANNEE",
        max_par_pilier=args.max_par_pilier,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_csv(args.out, index=False)
    print(f"[explore_embeddings] Fichier généré → {args.out}")


if __name__ == "__main__":
    main()
