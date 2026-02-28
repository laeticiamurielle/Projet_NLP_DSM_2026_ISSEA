"""
audit_snd30.nlp.classification
===============================
ÉTAPE 2 — Classification des libellés dans les piliers SND30
via CamemBERT (zero-shot puis fine-tuning).

Pipeline
--------
  1. zero_shot()  → pseudo-labels sans données annotées
  2. fine_tuner() → spécialisation sur le vocabulaire budgétaire
  3. predire()    → inférence finale sur les deux LF
"""

from __future__ import annotations

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    pipeline,
)
from datasets import Dataset

from src.config import (
    PILIERS, LABEL2ID, ID2LABEL,
    ZEROSHOT_MODEL, FINETUNE_MODEL, NLI_TEMPLATE, MODEL_DIR,
    MAX_LEN, TRAIN_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_RATIO, EARLY_STOP_PAT,
    SEUIL_ZEROSHOT, SEUIL_PRED, MIN_CONFIANCE_FT,
)


# ── Utilitaire interne ────────────────────────────────────────────────────────
def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _col_prob(pilier: str) -> str:
    """Nom de colonne de probabilité pour un pilier donné."""
    return "PROB_" + pilier.replace(" ", "_")[:12].upper()


# ── ÉTAPE 2a : Zero-shot ──────────────────────────────────────────────────────
def zero_shot(
    df: pd.DataFrame,
    libelle_col: str = "LIBELLE",
    seuil: float = SEUIL_ZEROSHOT,
) -> pd.DataFrame:
    """
    Classifie chaque libellé sans aucune donnée annotée.
    Utilise XLM-RoBERTa-XNLI via Hugging Face pipeline.

    Logique NLI
    -----------
    Le modèle transforme la classification en question d'inférence :
    "Cette ligne appartient au pilier X ?" → entailment / contradiction.
    Le pilier avec le score d'entailment le plus élevé est retenu.

    Paramètres
    ----------
    df          : DataFrame brut avec colonne LIBELLE
    libelle_col : nom de la colonne texte
    seuil       : confiance minimale ; en dessous → A_VERIFIER=True

    Retour
    ------
    df enrichi avec PILIER_SND30, CONFIANCE, A_VERIFIER, PROB_*
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"[zero_shot] Chargement {ZEROSHOT_MODEL} — {'GPU' if device == 0 else 'CPU'}")

    clf    = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=device)
    textes = df[libelle_col].fillna("").astype(str).tolist()

    resultats = []
    for i in tqdm(range(0, len(textes), 16), desc="Zero-shot"):
        batch   = textes[i: i + 16]
        sorties = clf(batch, candidate_labels=PILIERS,
                      hypothesis_template=NLI_TEMPLATE, multi_label=False)
        if isinstance(sorties, dict):
            sorties = [sorties]
        resultats.extend(sorties)

    df = df.copy()
    df["PILIER_SND30"] = [r["labels"][0]          for r in resultats]
    df["CONFIANCE"]    = [round(r["scores"][0], 4) for r in resultats]
    df["A_VERIFIER"]   = df["CONFIANCE"] < seuil

    for p in PILIERS:
        df[_col_prob(p)] = [
            round(dict(zip(r["labels"], r["scores"])).get(p, 0), 4)
            for r in resultats
        ]

    print(f"\n[zero_shot] Terminé — {df['A_VERIFIER'].sum()} lignes à vérifier")
    print(df["PILIER_SND30"].value_counts().to_string())
    return df


# ── ÉTAPE 2b : Fine-tuning ────────────────────────────────────────────────────
def fine_tuner(
    df_zeroshot: pd.DataFrame,
    min_confiance: float = MIN_CONFIANCE_FT,
) -> dict:
    """
    Fine-tune camembert-base sur les pseudo-labels fiables du zero-shot.

    Démarche
    --------
    1. Sélection des lignes avec confiance >= min_confiance
    2. Équilibrage des classes (sur-échantillonnage plafonné)
    3. Split stratifié train/eval (85/15)
    4. Fine-tuning avec scheduler cosine + early stopping
    5. Évaluation : F1-macro, F1-weighted, Log-Loss, Accuracy

    Paramètres
    ----------
    df_zeroshot   : sortie de zero_shot()
    min_confiance : seuil pour conserver un pseudo-label (défaut 0.60)

    Retour
    ------
    dict : f1_macro, f1_weighted, f1_par_pilier, log_loss,
           accuracy, classification_report
    """
    df_ok = df_zeroshot[df_zeroshot["CONFIANCE"] >= min_confiance].copy()
    df_ok["label"] = df_ok["PILIER_SND30"].map(LABEL2ID)

    print(f"\n[fine_tuner] {len(df_ok)}/{len(df_zeroshot)} lignes retenues "
          f"(confiance ≥ {min_confiance})")

    # Équilibrage
    plafond = df_ok["label"].value_counts().min() * 2
    df_ok = (df_ok.groupby("label", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), plafond), random_state=42)))

    df_train, df_eval = train_test_split(
        df_ok, test_size=0.15, stratify=df_ok["label"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)

    def tokenise(exemples):
        return tokenizer(exemples["LIBELLE"], truncation=True,
                         max_length=MAX_LEN, padding=False)

    def vers_hf(df_):
        ds = Dataset.from_dict({
            "LIBELLE": df_["LIBELLE"].astype(str).tolist(),
            "label":   df_["label"].astype(int).tolist(),
        })
        return ds.map(tokenise, batched=True, remove_columns=["LIBELLE"])

    train_ds = vers_hf(df_train)
    eval_ds  = vers_hf(df_eval)

    model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNE_MODEL,
        num_labels=len(PILIERS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def metriques(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = _softmax(logits)
        return {
            "f1_macro":    f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "log_loss":    log_loss(labels, probs),
            "accuracy":    accuracy_score(labels, preds),
        }

    args = TrainingArguments(
        output_dir                  = str(MODEL_DIR),
        num_train_epochs            = TRAIN_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        fp16                        = torch.cuda.is_available(),
        logging_steps               = 20,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        tokenizer       = tokenizer,
        data_collator   = DataCollatorWithPadding(tokenizer),
        compute_metrics = metriques,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    print("\n[fine_tuner] Démarrage fine-tuning CamemBERT...")
    trainer.train()

    pred_out = trainer.predict(eval_ds)
    preds    = np.argmax(pred_out.predictions, axis=-1)
    probs    = _softmax(pred_out.predictions)
    labels   = pred_out.label_ids

    scores = {
        "f1_macro":              round(f1_score(labels, preds, average="macro"),    4),
        "f1_weighted":           round(f1_score(labels, preds, average="weighted"), 4),
        "f1_par_pilier":         dict(zip(PILIERS, f1_score(labels, preds, average=None).round(4).tolist())),
        "log_loss":              round(log_loss(labels, probs), 4),
        "accuracy":              round(accuracy_score(labels, preds), 4),
        "classification_report": classification_report(labels, preds, target_names=PILIERS),
    }

    print("\n=== Scores de performance ===")
    print(f"  F1 macro    : {scores['f1_macro']}")
    print(f"  F1 weighted : {scores['f1_weighted']}")
    print(f"  Log-Loss    : {scores['log_loss']}")
    print(f"  Accuracy    : {scores['accuracy']}")
    print(scores["classification_report"])

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"[fine_tuner] Modèle sauvegardé → {MODEL_DIR}")
    return scores


# ── ÉTAPE 2c : Prédiction finale ──────────────────────────────────────────────
def predire(
    df: pd.DataFrame,
    libelle_col: str = "LIBELLE",
    seuil: float = SEUIL_PRED,
    model_dir: str | None = None,
) -> pd.DataFrame:
    """
    Inférence avec le modèle CamemBERT fine-tuné.

    À appeler sur df_2024 ET df_2025 après fine-tuning.
    L'année est propagée automatiquement (colonne ANNEE déjà présente).

    Paramètres
    ----------
    df          : DataFrame brut (avec colonne LIBELLE)
    libelle_col : nom de la colonne texte
    seuil       : seuil de confiance (A_VERIFIER si en dessous)
    model_dir   : chemin du modèle (défaut : MODEL_DIR du config)

    Retour
    ------
    df enrichi avec PILIER_SND30, CONFIANCE, A_VERIFIER, PROB_*
    """
    model_dir = model_dir or str(MODEL_DIR)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    textes    = df[libelle_col].fillna("").astype(str).tolist()
    resultats = []

    for i in tqdm(range(0, len(textes), 32), desc="Prédiction finale"):
        enc = tokenizer(
            textes[i: i + 32], truncation=True,
            max_length=MAX_LEN, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for p in probs:
            best = int(p.argmax())
            resultats.append({
                "pilier":    ID2LABEL[best],
                "confiance": float(p[best]),
                "probs":     {ID2LABEL[j]: float(p[j]) for j in range(len(PILIERS))},
            })

    df = df.copy()
    df["PILIER_SND30"] = [r["pilier"]             for r in resultats]
    df["CONFIANCE"]    = [round(r["confiance"], 4) for r in resultats]
    df["A_VERIFIER"]   = df["CONFIANCE"] < seuil
    for p in PILIERS:
        df[_col_prob(p)] = [round(r["probs"][p], 4) for r in resultats]
    return df
