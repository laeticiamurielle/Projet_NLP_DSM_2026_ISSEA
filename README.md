# 📊 Audit SND30 — Intelligence Artificielle et Finances Publiques

**ISE3-DS · ISSEA · Yaoundé — Année académique 2025-2026**

> *Comment l'IA peut-elle mesurer mathématiquement l'évolution des priorités de l'État camerounais entre la Loi de Finances 2024 et les perspectives de 2025-2026 ? Existe-t-il un alignement statistiquement significatif entre le discours budgétaire et les piliers de la SND30 ?*

---

## Architecture du projet

```
audit_snd30/
├── pyproject.toml                  # Configuration Poetry
├── README.md
├── .gitignore
│
├── audit_snd30/                    # Package Python principal
│   ├── config.py                   # ⚙️  Constantes centralisées (piliers, chemins, hyperparamètres)
│   │
│   ├── extraction/                 # 📄 Étape 1 : Extraction PDF
│   │   ├── base.py                 #    Utilitaires partagés (OCR, parse_amount, export Excel)
│   │   ├── lf_2024.py              #    Extracteur LF 2023-2024
│   │   └── lf_2025.py              #    Extracteur LF 2024-2025
│   │
│   ├── nlp/                        # 🤖 Étape 2 : Classification NLP
│   │   └── classification.py       #    Zero-shot · Fine-tuning · Prédiction (CamemBERT)
│   │
│   ├── analysis/                   # 📐 Étapes 3 & 4 : Analyse statistique
│   │   ├── glissement.py           #    Jensen-Shannon · TF-IDF Cosinus · Δ AE/CP
│   │   └── alignement.py           #    Test du Chi² vs cibles SND30
│   │
│   └── dashboard/                  # 🖥️  Tableau de bord Streamlit
│       └── app.py                  #    5 pages · Baromètre JS · Rapport technique
│
├── scripts/
│   ├── run_pipeline.py             # 🚀 Pipeline complet (extraction → analyse)
│   └── run_dashboard.py            # 🚀 Lancement dashboard
│
├── tests/
│   └── test_analysis.py            # ✅ Tests unitaires reproductibles (pytest)
│
└── data/
    ├── raw/                        # PDFs originaux MINFI
    └── processed/                  # CSV/Excel extraits et classifiés
```

---

## Installation

### Prérequis
- Python 3.10 ou 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Cloner le dépôt
git clone https://github.com/votre-groupe/audit-snd30.git
cd audit-snd30

# Installer les dépendances avec Poetry
poetry install

# Activer l'environnement virtuel
poetry shell
```

---

## Utilisation

### 1. Pipeline complet (extraction → classification → analyse)

```bash
# Avec les PDFs du MINFI
poetry run python scripts/run_pipeline.py \
    --pdf-2024 /chemin/vers/LF_2023-2024.pdf \
    --pdf-2025 /chemin/vers/LF_2024-2025.pdf

# Si l'extraction a déjà été faite (données dans data/processed/)
poetry run python scripts/run_pipeline.py --skip-extraction

# Zero-shot uniquement (sans fine-tuning, plus rapide)
poetry run python scripts/run_pipeline.py --skip-extraction --skip-finetuning
```

### 2. Extraction seule (CLI)

```bash
# Commande enregistrée par Poetry
snd30-extract --lf 2024 --pdf /chemin/LF_2023-2024.pdf
snd30-extract --lf 2025 --pdf /chemin/LF_2024-2025.pdf
```

### 3. Dashboard interactif

```bash
poetry run streamlit run audit_snd30/dashboard/app.py
```
→ Ouvrir http://localhost:8501

### 4. Tests unitaires

```bash
poetry run pytest tests/ -v

# Avec rapport de couverture
poetry run pytest tests/ -v --cov=audit_snd30 --cov-report=html
```

---

## Pipeline NLP — Détail méthodologique

```
PDF MINFI
   │
   ▼ Étape 1 — Extraction (pdfplumber)
┌──────────────────────────────────────┐
│  Détection des pages budgétaires     │
│  Extraction CODE · LIBELLE · AE · CP │
│  Corrections artefacts OCR           │
└──────────────────────────────────────┘
   │
   ▼ Étape 2 — Classification NLP (CamemBERT)
┌──────────────────────────────────────┐
│  2a. Zero-shot (XLM-RoBERTa-XNLI)   │
│      → Pseudo-labels sans annotation │
│  2b. Fine-tuning (camembert-base)    │
│      → Spécialisation vocabulaire    │
│         budgétaire camerounais       │
│  2c. Prédiction finale               │
│      → PILIER_SND30 · CONFIANCE      │
└──────────────────────────────────────┘
   │
   ▼ Étape 3 — Glissement Sémantique
┌──────────────────────────────────────┐
│  Jensen-Shannon (distributions)      │
│  TF-IDF Cosinus (vocabulaire)        │
│  Δ Part AE/CP (ressources)           │
└──────────────────────────────────────┘
   │
   ▼ Étape 4 — Alignement Statistique
┌──────────────────────────────────────┐
│  Test du Chi² vs cibles SND30        │
│  H₀ : budget aligné (p > 0.05)       │
│  H₁ : désaligné (p < 0.05)           │
└──────────────────────────────────────┘
```

*Projet ISE3-DS · ISSEA Yaoundé · 2025-2026*
