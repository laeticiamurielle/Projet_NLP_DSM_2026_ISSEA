"""
audit_snd30/config.py
=====================
Point unique de configuration du projet.
Toutes les constantes et chemins sont définis ici.
"""

from pathlib import Path
from typing import Final

# ── Racine du projet ──────────────────────────────────────────────────────────
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DIR:  Final[Path] = DATA_DIR / "raw"
PROC_DIR: Final[Path] = DATA_DIR / "processed"
MODEL_DIR: Final[Path] = ROOT_DIR / "modele_snd30"

# ── Piliers SND30 ─────────────────────────────────────────────────────────────
PILIERS: Final[list[str]] = [
    "Transformation Structurelle",
    "Capital Humain",
    "Gouvernance",
    "Développement Régional",
]

# Cibles théoriques de répartition SND30 (document officiel 2020-2030)
CIBLES_SND30: Final[dict[str, float]] = {
    "Transformation Structurelle": 0.35,
    "Capital Humain":              0.30,
    "Gouvernance":                 0.20,
    "Développement Régional":      0.15,
}

LABEL2ID: Final[dict[str, int]] = {p: i for i, p in enumerate(PILIERS)}
ID2LABEL: Final[dict[int, str]] = {i: p for i, p in enumerate(PILIERS)}

# ── Couleurs du dashboard ─────────────────────────────────────────────────────
COULEURS_PILIERS: Final[dict[str, str]] = {
    "Transformation Structurelle": "#1F77B4",
    "Capital Humain":              "#2CA02C",
    "Gouvernance":                 "#D62728",
    "Développement Régional":      "#FF7F0E",
}
COULEUR_2024: Final[str] = "#1F77B4"
COULEUR_2025: Final[str] = "#FF7F0E"

# ── Modèles NLP ───────────────────────────────────────────────────────────────
ZEROSHOT_MODEL: Final[str] = "joeddav/xlm-roberta-large-xnli"
FINETUNE_MODEL: Final[str] = "camembert-base"
NLI_TEMPLATE:   Final[str] = "Cette ligne de dépense budgétaire appartient au pilier {}."
MAX_LEN:        Final[int] = 128

# ── Hyperparamètres fine-tuning ───────────────────────────────────────────────
TRAIN_EPOCHS:      Final[int]   = 8
BATCH_SIZE:        Final[int]   = 16
LEARNING_RATE:     Final[float] = 2e-5
WEIGHT_DECAY:      Final[float] = 0.01
WARMUP_RATIO:      Final[float] = 0.1
EARLY_STOP_PAT:    Final[int]   = 3
MIN_CONFIANCE_FT:  Final[float] = 0.60   # seuil pour garder un pseudo-label
SEUIL_PRED:        Final[float] = 0.50   # seuil confiance prédiction finale
SEUIL_ZEROSHOT:    Final[float] = 0.45   # seuil confiance zero-shot

# ── Paramètres extraction LF 2024 ────────────────────────────────────────────
LF2024 = dict(
    x_no_max   = 90,
    x_code_min = 90,  x_code_max = 120,
    x_lib_min  = 120, x_lib_max  = 246,
    x_ae_min   = 430, x_ae_max   = 493,
    x_cp_min   = 493, x_cp_max   = 555,
    no_window  = 14,
    max_amount = 999_000_000,
    skip_pages = {108, 109, 110},
)

# ── Paramètres extraction LF 2025 ────────────────────────────────────────────
LF2025 = dict(
    budget_page_start  = 81,
    budget_page_end    = 106,
    ae_x_min = 425, ae_x_max = 478,
    cp_x_min = 480, cp_x_max = 536,
    libelle_col_start  = 17,
    libelle_col_end    = 42,
)


# ── Noms de fichiers PDF sources ──────────────────────────────────────────────
PDF_2024 = RAW_DIR / "LOI DES FINANCES 2023-2024.pdf"
PDF_2025 = RAW_DIR / "LOI DES FINANCES 2024-2025.pdf"

# ── Noms de fichiers de sortie ────────────────────────────────────────────────
FNAME_2024_RAW       = "lignes_budgetaires_2024"
FNAME_2025_RAW       = "lignes_budgetaires_2024_2025"
FNAME_2024_CLASSIFIE = "df_2024_classifie"
FNAME_2025_CLASSIFIE = "df_2025_classifie"
