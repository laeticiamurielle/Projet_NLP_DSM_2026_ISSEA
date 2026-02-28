"""
audit_snd30.extraction.lf_2024
===============================
Extracteur des lignes budgétaires depuis le PDF "LOI DES FINANCES 2023-2024.pdf" (MINFI).

Usage
-----
    from audit_snd30.extraction.lf_2024 import extraire_lf2024
    df = extraire_lf2024("data/raw/LOI DES FINANCES 2023-2024.pdf")
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

from audit_snd30.config import LF2024, PROC_DIR, FNAME_2024_RAW
from .base import clean_lib, digs, parse_amount, export_excel_formate


# ── Configuration (alias depuis config) ──────────────────────────────────────
X_NO_MAX    = LF2024["x_no_max"]
X_CODE_MIN  = LF2024["x_code_min"];  X_CODE_MAX  = LF2024["x_code_max"]
X_LIB_MIN   = LF2024["x_lib_min"];   X_LIB_MAX   = LF2024["x_lib_max"]
X_AE_MIN    = LF2024["x_ae_min"];    X_AE_MAX    = LF2024["x_ae_max"]
X_CP_MIN    = LF2024["x_cp_min"];    X_CP_MAX    = LF2024["x_cp_max"]
NO_WINDOW   = LF2024["no_window"]
MAX_AMOUNT  = LF2024["max_amount"]
SKIP_PAGES  = LF2024["skip_pages"]


# ── Helpers internes ──────────────────────────────────────────────────────────
def _zone(words: list, xmin: float, xmax: float) -> list:
    return [w for w in words if xmin <= w["x0"] < xmax]


def _txt(words: list) -> str:
    return " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"])).strip()


def _valid_ae(raw: str) -> bool:
    d = digs(raw)
    return 4 <= len(d) <= 12


def _is_chapitre(code_txt: str, lib_txt: str) -> bool:
    return "CHAPITRE" in (code_txt + " " + lib_txt).upper()


# ── Extraction par page ───────────────────────────────────────────────────────
def _extract_page(page) -> list[dict]:
    words = page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
    if not words:
        return []

    # Grouper par y (granularité 2 pts)
    row_map: dict[int, list] = {}
    for w in words:
        key = round(w["top"] / 2) * 2
        row_map.setdefault(key, []).append(w)
    sorted_ys = sorted(row_map.keys())

    # Identifier les pivots (lignes avec AE et CP valides)
    pivots = []
    for y in sorted_ys:
        rw = row_map[y]
        ae_raw = "".join(w["text"] for w in _zone(rw, X_AE_MIN, X_AE_MAX))
        cp_raw = "".join(w["text"] for w in _zone(rw, X_CP_MIN, X_CP_MAX))
        if _valid_ae(ae_raw) and _valid_ae(cp_raw):
            if _txt(_zone(rw, X_AE_MIN, X_AE_MAX)).upper() not in {"AE", "CP"}:
                pivots.append(y)

    if not pivots:
        return []

    results = []
    for i, piv_y in enumerate(pivots):
        piv_rw = row_map[piv_y]
        ae_d   = digs("".join(w["text"] for w in _zone(piv_rw, X_AE_MIN, X_AE_MAX)))
        cp_d   = digs("".join(w["text"] for w in _zone(piv_rw, X_CP_MIN, X_CP_MAX)))

        # Chercher N° et CODE dans la fenêtre ± NO_WINDOW
        no_text = code_text = ""
        for y2 in sorted_ys:
            if abs(y2 - piv_y) <= NO_WINDOW:
                rw2    = row_map[y2]
                no_str  = _txt(_zone(rw2, 0, X_NO_MAX)).strip()
                cod_str = _txt(_zone(rw2, X_CODE_MIN, X_CODE_MAX)).strip()
                if re.fullmatch(r"\d{1,3}", no_str) and int(no_str) <= 999:
                    no_text = no_str
                if re.fullmatch(r"\d{2,4}", cod_str):
                    code_text = cod_str

        # Bornes libellé
        y_lib_start = (pivots[i - 1] + NO_WINDOW + 2) if i > 0 else sorted_ys[0]
        y_lib_end   = piv_y + NO_WINDOW + 2

        lib_parts = []
        full_code_zone = ""
        for y2 in sorted_ys:
            if y_lib_start <= y2 <= y_lib_end:
                piece = _txt(_zone(row_map[y2], X_LIB_MIN, X_LIB_MAX))
                if piece:
                    lib_parts.append(piece)
                full_code_zone += " " + _txt(_zone(row_map[y2], X_CODE_MIN, X_CODE_MAX))

        libelle = clean_lib(" ".join(lib_parts))

        if _is_chapitre(full_code_zone, libelle):
            continue

        ae_val = int(ae_d) if ae_d else 0
        cp_val = int(cp_d) if cp_d else 0
        if ae_val > MAX_AMOUNT or cp_val > MAX_AMOUNT:
            continue
        if ae_val == 0 and cp_val == 0:
            continue

        results.append({
            "N°":                  no_text,
            "CODE":                code_text,
            "LIBELLE":             libelle or "(non extrait)",
            "AE":                  ae_val,
            "CP":                  cp_val,
            "_page":               page.page_number,
        })
    return results


# ── Détection des pages budgétaires ──────────────────────────────────────────
def _find_budget_pages(pdf) -> list[int]:
    found = []
    for i, page in enumerate(pdf.pages):
        if (i + 1) in SKIP_PAGES:
            continue
        word_set = {w["text"] for w in page.extract_words()}
        joined   = " ".join(word_set).upper()
        if "AE" in word_set and "CP" in word_set and (
            "LIBELLE" in word_set or "PROGRAMME" in joined
        ):
            found.append(i)
    return found


# ── Post-traitement ───────────────────────────────────────────────────────────
def _is_valid_row(row: dict) -> bool:
    code_text = str(row.get("CODE", "")).strip()
    libelle   = str(row.get("LIBELLE", "")).strip()
    if not re.fullmatch(r"\d{2,4}", code_text):
        return False
    lib_no_digits = re.sub(r"\d+", "", libelle).strip()
    return len(lib_no_digits) >= 5


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.apply(_is_valid_row, axis=1)].copy()
    df = df.drop_duplicates(subset=["CODE", "AE"], keep="first")
    df = df[df["AE"] > 0].reset_index(drop=True)
    df.index += 1
    return df


# ── Fonction principale exportée ──────────────────────────────────────────────
def extraire_lf2024(pdf_path: str, out_dir: Path = PROC_DIR) -> pd.DataFrame:
    """
    Extrait toutes les lignes budgétaires du PDF "LOI DES FINANCES 2023-2024.pdf".

    Paramètres
    ----------
    pdf_path : chemin absolu vers le PDF MINFI
    out_dir  : dossier de sortie pour Excel et CSV

    Retour
    ------
    DataFrame avec colonnes : N°, CODE, LIBELLE, AE, CP
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"[LF 2024] PDF ouvert : {len(pdf.pages)} pages")
        budget_pages = _find_budget_pages(pdf)
        print(f"[LF 2024] Pages analysées : {[p + 1 for p in budget_pages]}")

        for idx in budget_pages:
            rows = _extract_page(pdf.pages[idx])
            all_rows.extend(rows)
            if rows:
                print(f"  Page {idx + 1:>3} → {len(rows):>3} ligne(s)")

    cols = ["N°", "CODE", "LIBELLE", "AE", "CP", "_page"]
    df = pd.DataFrame(all_rows, columns=cols) if all_rows else pd.DataFrame(columns=cols)
    df = df.drop(columns=["_page"], errors="ignore")
    df = _clean_df(df)

    excel_path = out_dir / f"{FNAME_2024_RAW}.xlsx"
    csv_path   = out_dir / f"{FNAME_2024_RAW}.csv"
    export_excel_formate(df, excel_path, sheet_name="Lignes Budgétaires 2024")
    df.to_csv(str(csv_path), index=True, encoding="utf-8-sig")

    print(f"\n[LF 2024] ✓ {len(df)} lignes extraites")
    print(f"  Excel → {excel_path}")
    print(f"  CSV   → {csv_path}")
    return df
