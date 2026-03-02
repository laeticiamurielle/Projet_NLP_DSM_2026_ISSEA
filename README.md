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
│   │   ├── lf_2024.py              #    Extracteur LOI DES FINANCES 2023-2024
│   │   └── lf_2025.py              #    Extracteur LOI DES FINANCES 2024-2025
│   │
│   ├── nlp/                        # 🤖 Étape 2 : Classification NLP
│   │   ├── classification.py       #    Zero-shot · Fine-tuning · Prédiction (CamemBERT)
│   │   └── embeddings.py           #    SentenceTransformer (embeddings pour glissement & UMAP)
│   │
│   ├── analysis/                   # 📐 Étapes 3 & 4 : Analyse statistique
│   │   ├── glissement.py           #    Jensen-Shannon · TF-IDF Cosinus · Embeddings · Δ AE/CP
│   │   ├── embeddings_explorer.py  #    UMAP 2D pour explorer l'espace des embeddings
│   │   └── alignement.py           #    Test du Chi² vs cibles SND30
│   │
│   └── dashboard/                  # 🖥️  Tableau de bord Streamlit
│       └── app.py                  #    6 pages · Baromètre JS · Embeddings · Rapport technique
│
├── scripts/
│   ├── run_pipeline.py             # 🚀 Pipeline complet (extraction → analyse)
│   ├── run_dashboard.py            # 🚀 Lancement dashboard
│   └── explore_embeddings.py       # 🚀 Génération CSV UMAP des embeddings (analyse offline)
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
      --pdf-2024 data/raw/LOI DES FINANCES 2023-2024.pdf \
      --pdf-2025 data/raw/LOI DES FINANCES 2024-2025.pdf

# Si l'extraction a déjà été faite (données dans data/processed/)
poetry run python scripts/run_pipeline.py --skip-extraction

# Zero-shot uniquement (sans fine-tuning, plus rapide)
poetry run python scripts/run_pipeline.py --skip-extraction --skip-finetuning
```

### 2. Extraction seule (CLI)

```bash
# Commande enregistrée par Poetry
snd30-extract --lf 2024 --pdf data/raw/LOI DES FINANCES 2023-2024.pdf
snd30-extract --lf 2025 --pdf data/raw/LOI DES FINANCES 2024-2025.pdf
```

### 3. Dashboard interactif (avec page Embeddings)

```bash
poetry run streamlit run audit_snd30/dashboard/app.py
```
→ Ouvrir http://localhost:8501

Dans la barre latérale, la page **🧬 Embeddings** permet de :
- calculer les embeddings SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`),
- projeter les points en 2D via UMAP,
- visualiser l'espace sémantique par pilier SND30 et par année (2024 vs 2025).

### 3bis. Fichiers produits par le pipeline

Après exécution de `scripts/run_pipeline.py`, les principaux fichiers créés dans `data/processed/` sont :
- `lignes_budgetaires_2024.xlsx` : lignes budgétaires extraites de la LF 2024 (CODE, LIBELLE, AE, CP, ANNEE=2024)
- `lignes_budgetaires_2024_2025.xlsx` : lignes budgétaires extraites de la LF 2025 (CODE, LIBELLE, AE, CP, ANNEE=2025)
- `df_2024_classifie.xlsx` : lignes 2024 avec `PILIER_SND30`, `CONFIANCE` et probabilités par pilier
- `df_2025_classifie.xlsx` : lignes 2025 avec `PILIER_SND30`, `CONFIANCE` et probabilités par pilier
- éventuellement `embeddings_umap.csv` si vous lancez `scripts/explore_embeddings.py` (projection 2D pour analyses externes)

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
│  2a. Zero-shot (MiniLM multilingue) │
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
│  Embeddings SentenceTransformer +    │
│    UMAP 2D (glissement sémantique)   │
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
