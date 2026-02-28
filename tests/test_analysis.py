"""
tests/test_analysis.py
=======================
Tests unitaires pour les modules d'analyse.
Reproductibles sans GPU ni PDF (données synthétiques).

Lancement :
    poetry run pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from audit_snd30.config import PILIERS, CIBLES_SND30
from audit_snd30.analysis.glissement import calculer_glissement
from audit_snd30.analysis.alignement import test_alignement


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def df_synth_2024() -> pd.DataFrame:
    """DataFrame synthétique simulant une LF 2024 classifiée."""
    rng = np.random.default_rng(42)
    n   = 120
    piliers = rng.choice(PILIERS, size=n, p=[0.35, 0.30, 0.20, 0.15])
    return pd.DataFrame({
        "LIBELLE":     [f"Programme de développement {i}" for i in range(n)],
        "AE":          rng.integers(500_000, 50_000_000, size=n),
        "CP":          rng.integers(500_000, 50_000_000, size=n),
        "PILIER_SND30": piliers,
        "CONFIANCE":   rng.uniform(0.5, 0.99, size=n).round(4),
        "A_VERIFIER":  [False] * n,
        "ANNEE":       [2024] * n,
    })


@pytest.fixture
def df_synth_2025(df_synth_2024) -> pd.DataFrame:
    """DataFrame 2025 : légèrement décalé par rapport à 2024."""
    rng = np.random.default_rng(99)
    n   = 115
    # Distribution légèrement différente pour simuler un glissement
    piliers = rng.choice(PILIERS, size=n, p=[0.40, 0.28, 0.18, 0.14])
    df = df_synth_2024.iloc[:n].copy()
    df["PILIER_SND30"] = piliers
    df["ANNEE"]        = 2025
    df["AE"]           = rng.integers(500_000, 55_000_000, size=n)
    df["CP"]           = rng.integers(500_000, 55_000_000, size=n)
    return df


# ── Tests glissement ──────────────────────────────────────────────────────────
class TestGlissement:

    def test_retourne_les_bonnes_cles(self, df_synth_2024, df_synth_2025):
        g = calculer_glissement(df_synth_2024, df_synth_2025)
        cles_attendues = {
            "distributions", "jensen_shannon", "jensen_shannon_interpretation",
            "cosine_tfidf", "parts_ae", "parts_cp", "delta_ae", "delta_cp",
            "montants_ae", "montants_cp",
        }
        assert cles_attendues.issubset(g.keys())

    def test_jensen_shannon_entre_0_et_1(self, df_synth_2024, df_synth_2025):
        g = calculer_glissement(df_synth_2024, df_synth_2025)
        assert 0.0 <= g["jensen_shannon"] <= 1.0

    def test_jensen_shannon_zero_si_distributions_identiques(self, df_synth_2024):
        g = calculer_glissement(df_synth_2024, df_synth_2024)
        assert g["jensen_shannon"] == pytest.approx(0.0, abs=1e-6)

    def test_delta_ae_somme_nulle_si_meme_df(self, df_synth_2024):
        """Si les deux années sont identiques, Δ AE doit être 0 pour chaque pilier."""
        g = calculer_glissement(df_synth_2024, df_synth_2024)
        for p in PILIERS:
            assert g["delta_ae"][p] == pytest.approx(0.0, abs=0.01)

    def test_distributions_somment_a_1(self, df_synth_2024, df_synth_2025):
        g = calculer_glissement(df_synth_2024, df_synth_2025)
        for annee in [2024, 2025]:
            total = sum(g["distributions"][annee].values())
            assert total == pytest.approx(1.0, abs=1e-4)

    def test_cosine_tfidf_piliers_presens(self, df_synth_2024, df_synth_2025):
        g = calculer_glissement(df_synth_2024, df_synth_2025)
        assert set(g["cosine_tfidf"].keys()) == set(PILIERS)


# ── Tests alignement ──────────────────────────────────────────────────────────
class TestAlignement:

    def test_retourne_deux_lignes(self, df_synth_2024, df_synth_2025):
        res = test_alignement(df_synth_2024, df_synth_2025)
        assert len(res) == 2
        assert set(res["ANNEE"]) == {2024, 2025}

    def test_colonnes_attendues(self, df_synth_2024, df_synth_2025):
        res = test_alignement(df_synth_2024, df_synth_2025)
        for col in ["ANNEE", "CHI2", "P_VALUE", "ALIGNEMENT", "INTERPRETATION", "ECARTS_PAR_PILIER"]:
            assert col in res.columns

    def test_chi2_positif(self, df_synth_2024, df_synth_2025):
        res = test_alignement(df_synth_2024, df_synth_2025)
        assert (res["CHI2"] >= 0).all()

    def test_p_value_entre_0_et_1(self, df_synth_2024, df_synth_2025):
        res = test_alignement(df_synth_2024, df_synth_2025)
        assert ((res["P_VALUE"] >= 0) & (res["P_VALUE"] <= 1)).all()

    def test_distribution_parfaite_donne_alignement(self, df_synth_2024):
        """Une distribution parfaitement alignée avec SND30 doit donner p > 0.05."""
        n = 1000
        piliers_parfaits = []
        for p, prop in CIBLES_SND30.items():
            piliers_parfaits.extend([p] * int(prop * n))
        df_parfait = pd.DataFrame({
            "LIBELLE":     [f"ligne {i}" for i in range(len(piliers_parfaits))],
            "AE":          [1_000_000] * len(piliers_parfaits),
            "CP":          [1_000_000] * len(piliers_parfaits),
            "PILIER_SND30": piliers_parfaits,
            "CONFIANCE":   [0.9] * len(piliers_parfaits),
            "A_VERIFIER":  [False] * len(piliers_parfaits),
            "ANNEE":       [2024] * len(piliers_parfaits),
        })
        res = test_alignement(df_parfait, df_parfait)
        # La distribution parfaite doit être alignée
        assert (res["P_VALUE"] > 0.05).all(), \
            f"Distribution parfaite devrait être alignée, p={res['P_VALUE'].tolist()}"

    def test_ecarts_par_pilier_est_dict(self, df_synth_2024, df_synth_2025):
        res = test_alignement(df_synth_2024, df_synth_2025)
        for _, row in res.iterrows():
            assert isinstance(row["ECARTS_PAR_PILIER"], dict)
            assert set(row["ECARTS_PAR_PILIER"].keys()) == set(PILIERS)


# ── Tests extraction (sans PDF) ───────────────────────────────────────────────
class TestExtractionBase:

    def test_parse_amount_simple(self):
        from audit_snd30.extraction.base import parse_amount
        assert parse_amount(["1", "234", "567"]) == 1234567

    def test_parse_amount_none_si_vide(self):
        from audit_snd30.extraction.base import parse_amount
        assert parse_amount([]) is None

    def test_clean_lib_supprime_espaces_multiples(self):
        from audit_snd30.extraction.base import clean_lib
        result = clean_lib("PROGRAMME   DE   DÉVELOPPEMENT")
        assert "  " not in result

    def test_clean_ocr_corrige_artefacts(self):
        from audit_snd30.extraction.base import clean_ocr
        assert "COORDINATION" in clean_ocr("COwORDINATION DES ACTIONS")
        assert "MULTILATÉRALE" in clean_ocr("MULTJLATERALE COOPERATION")
