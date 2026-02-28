"""
audit_snd30.extraction
======================
Sous-package d'extraction PDF des lignes budgétaires.

Point d'entrée CLI (enregistré dans pyproject.toml) :
    snd30-extract --lf 2024 --pdf /chemin/vers/loi.pdf
"""

from .base import parse_amount, clean_lib
from .lf_2024 import extraire_lf2024
from .lf_2025 import extraire_lf2025

import argparse
from pathlib import Path


def main_cli() -> None:
    """Point d'entrée `snd30-extract` enregistré par Poetry."""
    parser = argparse.ArgumentParser(
        description="Extraction des lignes budgétaires depuis un PDF Loi de Finances"
    )
    parser.add_argument("--lf",  required=True, choices=["2024", "2025"],
                        help="Loi de Finances à extraire (2024 ou 2025)")
    parser.add_argument("--pdf", required=True, type=Path,
                        help="Chemin vers le fichier PDF")
    parser.add_argument("--out", type=Path, default=None,
                        help="Dossier de sortie (défaut: data/processed/)")
    args = parser.parse_args()

    from audit_snd30.config import PROC_DIR

    out_dir = args.out or PROC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lf == "2024":
        df = extraire_lf2024(str(args.pdf), out_dir=out_dir)
    else:
        df = extraire_lf2025(str(args.pdf), out_dir=out_dir)

    print(f"\n✓ {len(df)} lignes extraites → {out_dir}")


__all__ = ["extraire_lf2024", "extraire_lf2025", "parse_amount", "clean_lib", "main_cli"]
