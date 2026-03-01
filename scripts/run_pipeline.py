"""
scripts/run_pipeline.py
=======================
Pipeline complet : extraction → classification → analyse → export.

Usage
-----
    poetry run python scripts/run_pipeline.py \
        --pdf-2024 data/raw/LOI DES FINANCES 2023-2024.pdf \
        --pdf-2025 data/raw/LOI DES FINANCES 2024-2025.pdf

    # Ou si les données sont déjà extraites (CSV/Excel) :
    poetry run python scripts/run_pipeline.py --skip-extraction
"""

import argparse
from pathlib import Path

import pandas as pd

from audit_snd30.config import PROC_DIR, MODEL_DIR, FNAME_2024_RAW, FNAME_2025_RAW, PDF_2024, PDF_2025
from audit_snd30.extraction.articles import articles_nettoyes
from audit_snd30.nlp.classification import fine_tuner, predire, zero_shot
from audit_snd30.analysis.glissement import calculer_glissement
from audit_snd30.analysis.alignement import test_alignement


def main():
    parser = argparse.ArgumentParser(description="Pipeline NLP SND30 complet")
    parser.add_argument("--pdf-2024", type=Path, default=None,
                        help="Chemin vers le PDF LOI DES FINANCES 2023-2024.pdf (défaut: data/raw/LOI DES FINANCES 2023-2024.pdf)")
    parser.add_argument("--pdf-2025", type=Path, default=None,
                        help="Chemin vers le PDF LOI DES FINANCES 2024-2025.pdf (défaut: data/raw/LOI DES FINANCES 2024-2025.pdf)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Sauter l'extraction si les CSV/Excel existent déjà")
    parser.add_argument("--skip-finetuning", action="store_true",
                        help="Utiliser uniquement le zero-shot (pas de fine-tuning)")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    sep = "=" * 65

    # ── ÉTAPE 1 : Extraction ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ÉTAPE 1 — Extraction des lignes budgétaires")
    print(sep)


    # Extraction et nettoyage des articles (nouvelle étape)
    p24_xlsx = PROC_DIR / f"articles_2023_2024.xlsx"
    p25_xlsx = PROC_DIR / f"articles_2024_2025.xlsx"

    if args.skip_extraction and p24_xlsx.exists() and p25_xlsx.exists():
        print("[INFO] Extraction ignorée — fichiers existants chargés")
        df_2024 = pd.read_excel(str(p24_xlsx))
        df_2025 = pd.read_excel(str(p25_xlsx))
    else:
        pdf_2024 = args.pdf_2024 if args.pdf_2024 is not None else PDF_2024
        pdf_2025 = args.pdf_2025 if args.pdf_2025 is not None else PDF_2025
        if not Path(pdf_2024).exists() or not Path(pdf_2025).exists():
            import os
            print(f"[DEBUG] Chemin courant: {os.getcwd()}")
            print(f"[DEBUG] pdf_2024 attendu: {pdf_2024}")
            print(f"[DEBUG] pdf_2025 attendu: {pdf_2025}")
            raise ValueError(
                f"Fichier PDF introuvable : {pdf_2024 if not Path(pdf_2024).exists() else pdf_2025}\n"
                "Vérifiez que les fichiers sont bien placés dans data/raw/ ou fournissez le chemin avec --pdf-2024 et --pdf-2025."
            )
        # Extraction et nettoyage des articles
        articles_2024 = articles_nettoyes(pdf_2024)
        articles_2025 = articles_nettoyes(pdf_2025)
        df_2024 = pd.DataFrame(articles_2024)
        df_2025 = pd.DataFrame(articles_2025)
        df_2024.to_excel(str(p24_xlsx), index=False)
        df_2025.to_excel(str(p25_xlsx), index=False)

    df_2024["ANNEE"] = 2024
    df_2025["ANNEE"] = 2025
    print(f"\n  Articles LF 2023-2024 : {len(df_2024)} extraits et nettoyés")
    print(f"  Articles LF 2024-2025 : {len(df_2025)} extraits et nettoyés")

    # ── ÉTAPE 2 : Classification NLP ────────────────────────────────────────
    print(f"\n{sep}")
    print("  ÉTAPE 2 — Classification NLP (CamemBERT)")
    print(sep)

    print("\n[2a] Zero-shot sur LF 2024...")
    df_2024_zs = zero_shot(df_2024, libelle_col="LIBELLE")
    print("\n[2a] Zero-shot sur LF 2025...")
    df_2025_zs = zero_shot(df_2025, libelle_col="LIBELLE")

    if not args.skip_finetuning:
        print("\n[2b] Fine-tuning CamemBERT sur LF 2024...")
        scores = fine_tuner(df_2024_zs)
        print(f"\n  F1 macro : {scores['f1_macro']}")
        print(f"  Log-Loss : {scores['log_loss']}")

        print("\n[2c] Prédiction finale...")
        df_2024_final = predire(df_2024)
        df_2025_final = predire(df_2025)
    else:
        print("[INFO] Fine-tuning ignoré — utilisation des prédictions zero-shot")
        df_2024_final = df_2024_zs
        df_2025_final = df_2025_zs

    # Sauvegarde des DataFrames classifiés
    df_2024_final.to_excel(str(PROC_DIR / "df_2024_classifie.xlsx"), index=False)
    df_2025_final.to_excel(str(PROC_DIR / "df_2025_classifie.xlsx"), index=False)

    # ── ÉTAPE 3 : Glissement sémantique ─────────────────────────────────────
    print(f"\n{sep}")
    print("  ÉTAPE 3 — Glissement Sémantique")
    print(sep)

    glissement = calculer_glissement(df_2024_final, df_2025_final)
    js = glissement["jensen_shannon"]
    print(f"\n  Jensen-Shannon 2024→2025 : {js}")
    print(f"  Interprétation : {glissement['jensen_shannon_interpretation']}")
    print("\n  Δ AE par pilier :")
    for p, delta in glissement["delta_ae"].items():
        sens = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"    {p:<35} : {delta:+.2f}% {sens}")

    # ── ÉTAPE 4 : Alignement Chi² ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ÉTAPE 4 — Test d'Alignement Statistique (Chi²)")
    print(sep)

    alignement = test_alignement(df_2024_final, df_2025_final)
    print("\n" + alignement[["ANNEE", "CHI2", "P_VALUE", "ALIGNEMENT"]].to_string(index=False))
    for _, row in alignement.iterrows():
        print(f"\n  {row['INTERPRETATION']}")

    print(f"\n{sep}")
    print("  ✓ Pipeline terminé avec succès")
    print(f"  Données → {PROC_DIR}")
    if not args.skip_finetuning:
        print(f"  Modèle  → {MODEL_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
