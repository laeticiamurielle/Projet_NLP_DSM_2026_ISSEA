"""
audit_snd30.dashboard.app
==========================
Tableau de bord Streamlit — Baromètre de Glissement Sémantique SND30.

Lancement
---------
    poetry run streamlit run audit_snd30/dashboard/app.py

Architecture des pages
----------------------
  🏠 Vue d'ensemble  — KPI + graphe comparatif 2024 vs 2025
  📡 Baromètre JS    — jauge Jensen-Shannon + cosinus TF-IDF
  📈 Évolution       — Δ AE/CP par pilier + tableaux
  🔬 Analyse / Année — zoom par année (pie + histogramme confiance)
  📋 Rapport         — architecture modèle + scores + alignement Chi²
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from audit_snd30.analysis.alignement import test_alignement
from audit_snd30.analysis.glissement import calculer_glissement
from audit_snd30.analysis.embeddings_explorer import construire_embeddings_df
from audit_snd30.config import (
    CIBLES_SND30,
    COULEUR_2024,
    COULEUR_2025,
    COULEURS_PILIERS,
    PILIERS,
    PROC_DIR,
    MODEL_DIR,
)
from audit_snd30.nlp.classification import predire, zero_shot

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(page_title="Baromètre SND30", page_icon="📊", layout="wide")


# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def charger_donnees():
    """
    Charge les deux DataFrames classifiés.

    Logique de priorité :
      1. Fichiers déjà classifiés (data/processed/) → chargement direct
      2. Fichiers bruts → classification automatique (zero-shot ou modèle)
    """
    p24 = PROC_DIR / "df_2024_classifie.xlsx"
    p25 = PROC_DIR / "df_2025_classifie.xlsx"

    if p24.exists() and p25.exists():
        df_2024 = pd.read_excel(str(p24))
        df_2025 = pd.read_excel(str(p25))
        df_2024["ANNEE"] = 2024
        df_2025["ANNEE"] = 2025
    else:
        # Fichiers bruts produits par la pipeline d'extraction
        raw_24 = PROC_DIR / "articles_2023_2024.xlsx"
        raw_25 = PROC_DIR / "articles_2024_2025.xlsx"

        if not raw_24.exists() or not raw_25.exists():
            raise FileNotFoundError(
                "Fichiers bruts introuvables. Lance d'abord 'poetry run python scripts/run_pipeline.py' "
                "ou fournis manuellement les fichiers articles_2023_2024.xlsx et articles_2024_2025.xlsx dans data/processed/."
            )

        df_2024 = pd.read_excel(str(raw_24));  df_2024["ANNEE"] = 2024
        df_2025 = pd.read_excel(str(raw_25));  df_2025["ANNEE"] = 2025

        # Harmonisation du nom de la colonne texte pour la classification
        for df in (df_2024, df_2025):
            if "LIBELLE" not in df.columns and "texte_nettoye" in df.columns:
                df["LIBELLE"] = df["texte_nettoye"]

        with st.spinner("Classification CamemBERT en cours..."):
            if MODEL_DIR.exists():
                df_2024 = predire(df_2024)
                df_2025 = predire(df_2025)
            else:
                # On précise explicitement la colonne texte au cas où
                # le renommage précédent n'aurait pas eu lieu.
                df_2024 = zero_shot(df_2024, libelle_col="LIBELLE")
                df_2025 = zero_shot(df_2025, libelle_col="LIBELLE")

        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df_2024.to_excel(str(p24), index=False)
        df_2025.to_excel(str(p25), index=False)

    glissement = calculer_glissement(df_2024, df_2025)
    alignement = test_alignement(df_2024, df_2025)
    return df_2024, df_2025, glissement, alignement


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.title("📌 Navigation")
    annee_sel = st.sidebar.radio("Année à analyser", [2024, 2025, "2024 vs 2025"], index=2)
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Page", [
        "🏠 Vue d'ensemble",
        "📡 Baromètre JS",
        "📈 Évolution budgétaire",
        "🔬 Analyse par année",
        "🧬 Embeddings",
        "📋 Rapport technique",
    ])
    st.sidebar.markdown("---")
    st.sidebar.caption("Analyse NLP — évolution des priorités budgétaires camerounaises 2024 → 2025 via CamemBERT")
    return page, annee_sel


# ── Page 1 : Vue d'ensemble ───────────────────────────────────────────────────
def page_vue_ensemble(df_2024, df_2025, glissement, alignement):
    st.title("📊 Baromètre de Glissement Sémantique SND30")
    st.caption("Mesure de l'évolution des priorités budgétaires camerounaises — LF 2024 → LF 2025")
    st.markdown("---")

    js = glissement["jensen_shannon"]
    emoji_js = "🟢" if js < 0.10 else "🟡" if js < 0.25 else "🔴"
    row_2024 = alignement[alignement["ANNEE"] == 2024].iloc[0]
    row_2025 = alignement[alignement["ANNEE"] == 2025].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Lignes LF 2024", f"{len(df_2024):,}")
    col2.metric("Lignes LF 2025", f"{len(df_2025):,}", delta=f"{len(df_2025)-len(df_2024):+,}")
    col3.metric("Glissement JS", f"{js:.4f}", help="0=stable / 1=rupture totale")
    col3.caption(f"{emoji_js} {glissement['jensen_shannon_interpretation']}")
    col4.metric("Alignement 2024", row_2024["ALIGNEMENT"], help=f"p={row_2024['P_VALUE']:.3f}")
    col5.metric("Alignement 2025", row_2025["ALIGNEMENT"], help=f"p={row_2025['P_VALUE']:.3f}")
    st.markdown("---")

    parts_ae = glissement["parts_ae"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="LF 2024", x=PILIERS,
                         y=[parts_ae[2024][p] for p in PILIERS], marker_color=COULEUR_2024,
                         text=[f"{parts_ae[2024][p]:.1f}%" for p in PILIERS], textposition="outside"))
    fig.add_trace(go.Bar(name="LF 2025", x=PILIERS,
                         y=[parts_ae[2025][p] for p in PILIERS], marker_color=COULEUR_2025,
                         text=[f"{parts_ae[2025][p]:.1f}%" for p in PILIERS], textposition="outside"))
    fig.add_trace(go.Scatter(name="Cible SND30", x=PILIERS,
                             y=[CIBLES_SND30[p] * 100 for p in PILIERS],
                             mode="markers+lines",
                             marker=dict(size=12, symbol="diamond", color="black"),
                             line=dict(color="black", dash="dash")))
    fig.update_layout(title="Part des AE par pilier SND30 — 2024 vs 2025 vs Cibles",
                      barmode="group", yaxis_title="Part des AE (%)",
                      xaxis_tickangle=-15, height=450,
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)


# ── Page 2 : Baromètre JS ────────────────────────────────────────────────────
def page_barometre(glissement):
    st.title("📡 Baromètre Jensen-Shannon")
    js = glissement["jensen_shannon"]
    couleur_jauge = "#27AE60" if js < 0.10 else "#F39C12" if js < 0.25 else "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=js,
        number={"valueformat": ".4f"},
        title={"text": "Glissement Sémantique 2024 → 2025"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar":  {"color": couleur_jauge},
            "steps": [
                {"range": [0.00, 0.10], "color": "#D5F5E3"},
                {"range": [0.10, 0.25], "color": "#FCF3CF"},
                {"range": [0.25, 1.00], "color": "#FADBD8"},
            ],
        },
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Interprétation** : {glissement['jensen_shannon_interpretation']}")
    st.markdown("---")

    st.subheader("Similarité vocabulaire par pilier (TF-IDF Cosinus)")
    cosine = glissement["cosine_tfidf"]
    for pilier, score in cosine.items():
        if score is not None:
            st.metric(pilier, f"{score:.4f}", help="1 = même vocabulaire / 0 = vocabulaire totalement différent")


# ── Page 3bis : Embeddings SentenceTransformer ──────────────────────────────
def page_embeddings(df_2024, df_2025):
    st.title("🧬 Exploration des Embeddings (SentenceTransformer + UMAP)")
    st.caption("Projection 2D des libellés budgétaires, colorée par pilier et année")

    max_par_pilier = st.slider(
        "Nombre maximal de lignes par pilier et par année",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
    )

    df_all = pd.concat([df_2024, df_2025], ignore_index=True)

    with st.spinner("Calcul des embeddings et de l'UMAP 2D..."):
        df_emb = construire_embeddings_df(
            df_all,
            text_col="LIBELLE",
            label_col="PILIER_SND30",
            year_col="ANNEE",
            max_par_pilier=max_par_pilier,
        )

    col1, col2 = st.columns(2)
    with col1:
        fig_pilier = px.scatter(
            df_emb,
            x="x",
            y="y",
            color="PILIER_SND30",
            hover_data=["ANNEE", "LIBELLE"],
            color_discrete_map=COULEURS_PILIERS,
            title="Espace sémantique par pilier",
            height=500,
        )
        st.plotly_chart(fig_pilier, use_container_width=True)

    with col2:
        fig_annee = px.scatter(
            df_emb,
            x="x",
            y="y",
            color="ANNEE",
            hover_data=["PILIER_SND30", "LIBELLE"],
            color_discrete_sequence=[COULEUR_2024, COULEUR_2025],
            title="Espace sémantique par année",
            height=500,
        )
        st.plotly_chart(fig_annee, use_container_width=True)

    st.markdown("---")
    st.subheader("Aperçu des points (échantillon)")
    st.dataframe(df_emb.sample(min(200, len(df_emb))).reset_index(drop=True), use_container_width=True)


# ── Page 3 : Évolution budgétaire ────────────────────────────────────────────
def page_evolution(glissement):
    st.title("📈 Évolution Budgétaire 2024 → 2025")
    delta_ae = glissement["delta_ae"]

    fig = go.Figure(go.Bar(
        x=list(delta_ae.keys()),
        y=list(delta_ae.values()),
        marker_color=["#27AE60" if v >= 0 else "#E74C3C" for v in delta_ae.values()],
        text=[f"{v:+.2f}%" for v in delta_ae.values()],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title="Glissement budgétaire par pilier (Δ AE % — 2024 → 2025)",
                      yaxis_title="Variation (points de %)", xaxis_tickangle=-15, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ── Page 4 : Analyse par année ────────────────────────────────────────────────
def page_analyse_annee(df_2024, df_2025, annee_sel):
    st.title("🔬 Analyse par Année")
    dfs = {2024: df_2024, 2025: df_2025}
    annees = [2024, 2025] if annee_sel == "2024 vs 2025" else [annee_sel]

    for annee in annees:
        df = dfs[annee]
        couleur = COULEUR_2024 if annee == 2024 else COULEUR_2025
        st.subheader(f"LF {annee}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre de lignes", f"{len(df):,}")
        col2.metric("Total AE", f"{df['AE'].sum():,.0f}")
        col3.metric("Lignes à vérifier", f"{df['A_VERIFIER'].sum()}", help="Confiance < 0.50")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(df, names="PILIER_SND30",
                             title=f"Répartition par pilier — LF {annee}",
                             color="PILIER_SND30", color_discrete_map=COULEURS_PILIERS, height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            fig_hist = px.histogram(df, x="CONFIANCE", nbins=20,
                                    title=f"Distribution des scores de confiance — LF {annee}",
                                    color_discrete_sequence=[couleur], height=350)
            fig_hist.add_vline(x=0.50, line_dash="dash", line_color="red", annotation_text="Seuil 0.50")
            st.plotly_chart(fig_hist, use_container_width=True)

        pilier_choisi = st.selectbox(f"Voir les libellés du pilier ({annee})", PILIERS, key=f"pilier_{annee}")
        df_p = (df[df["PILIER_SND30"] == pilier_choisi]
                .nlargest(10, "AE")[["LIBELLE", "AE", "CP", "CONFIANCE"]].reset_index(drop=True))
        st.dataframe(df_p, use_container_width=True, hide_index=True)
        st.markdown("---")


# ── Page 5 : Rapport technique ────────────────────────────────────────────────
def page_rapport(glissement, alignement):
    st.title("📋 Rapport Technique")

    st.subheader("1. Architecture CamemBERT")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Paramètre | Valeur |
        |---|---|
        | Modèle zero-shot | xlm-roberta-large-xnli |
        | Modèle fine-tuning | camembert-base |
        | Couches Transformer | 12 |
        | Dimension cachée | 768 |
        | Paramètres | ~111M |
        | Tokenizer | SentencePiece 32k |
        """)
    with col2:
        st.markdown("""
        | Hyperparamètre | Valeur |
        |---|---|
        | Learning rate | 2e-5 |
        | Scheduler | Cosine |
        | Epochs max | 8 |
        | Early stopping | patience=3 |
        | Batch size | 16 |
        | Max tokens | 128 |
        """)

    st.markdown("---")
    st.subheader("2. Glissement Sémantique Jensen-Shannon")
    js = glissement["jensen_shannon"]
    st.markdown(f"""
    - **Période** : LF 2024 → LF 2025
    - **Score JS** : `{js}`
    - **Interprétation** : {glissement['jensen_shannon_interpretation']}
    - **Formule** : JS(P||Q) = 0.5·KL(P||M) + 0.5·KL(Q||M)
    - **Seuils** : < 0.10 stable · 0.10–0.25 modéré · > 0.25 fort
    """)

    st.markdown("---")
    st.subheader("3. Alignement Statistique (Chi²)")
    st.markdown("**H₀** : budget aligné avec cibles SND30 | **décision** : p < 0.05 → désalignement")
    for _, row in alignement.iterrows():
        fn = st.success if row["P_VALUE"] > 0.05 else st.error
        fn(f"**LF {int(row['ANNEE'])}** — χ²={row['CHI2']} | p={row['P_VALUE']} | {row['ALIGNEMENT']}  \n{row['INTERPRETATION']}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Chargement et classification en cours..."):
        df_2024, df_2025, glissement, alignement = charger_donnees()

    page, annee_sel = sidebar()

    dispatch = {
        "🏠 Vue d'ensemble":    lambda: page_vue_ensemble(df_2024, df_2025, glissement, alignement),
        "📡 Baromètre JS":      lambda: page_barometre(glissement),
        "📈 Évolution budgétaire": lambda: page_evolution(glissement),
        "🔬 Analyse par année": lambda: page_analyse_annee(df_2024, df_2025, annee_sel),
        "🧬 Embeddings":        lambda: page_embeddings(df_2024, df_2025),
        "📋 Rapport technique": lambda: page_rapport(glissement, alignement),
    }
    dispatch[page]()


if __name__ == "__main__":
    main()
