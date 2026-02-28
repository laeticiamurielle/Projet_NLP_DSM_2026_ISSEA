"""
audit_snd30.extraction.lf_2025
===============================
Extracteur des lignes budgétaires depuis la Loi de Finances 2024-2025 (PDF MINFI).

Usage
-----
    from audit_snd30.extraction.lf_2025 import extraire_lf2025
    df = extraire_lf2025("/chemin/vers/LF_2024-2025.pdf")
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

from src.config import LF2025, PROC_DIR, FNAME_2025_RAW
from .base import clean_ocr, parse_amount, export_excel_formate


# ── Configuration ─────────────────────────────────────────────────────────────
PAGE_START       = LF2025["budget_page_start"]
PAGE_END         = LF2025["budget_page_end"]
AE_X_MIN         = LF2025["ae_x_min"];  AE_X_MAX = LF2025["ae_x_max"]
CP_X_MIN         = LF2025["cp_x_min"];  CP_X_MAX = LF2025["cp_x_max"]
LIBELLE_COL_ST   = LF2025["libelle_col_start"]
LIBELLE_COL_END  = LF2025["libelle_col_end"]

LIBELLE_BLACKLIST = [
    "CODE LIBELLE", "PROGRAMME OBJEC", "No CODE",
    "ARTICLE SOIXANTE", "ARTICLE QUATRE", "TOTAL2025",
]

CHAPITRE_RE = re.compile(r"CHAPITRE", re.IGNORECASE)
ANCHOR_RE   = re.compile(r"^\s{3,16}(\d{1,3})?\s+(\d{3})\s+\S")


# ── Helpers internes ──────────────────────────────────────────────────────────
def _extract_libelle_from_line(line: str) -> str:
    if not line:
        return ""
    zone     = line[LIBELLE_COL_ST: min(LIBELLE_COL_END, len(line))]
    pre_zone = line[:LIBELLE_COL_ST] if len(line) >= LIBELLE_COL_ST else line
    pre_words = re.findall(r"[A-ZÀÂÉÈÊÙÛÔÎŒÆ][A-ZÀÂÉÈÊÙÛÔÎŒÆ\-\']{2,}", pre_zone)
    full_text = " ".join(pre_words) + " " + zone
    tokens = full_text.split()
    libelle_tokens = []
    for token in tokens:
        token = re.sub(r"^[wcmrp]+([A-ZÀÂÉÈÊÙÛÔÎŒÆ])", r"\1", token)
        clean = re.sub(r"['\u2019\u2018\-]", "", token)
        if not clean or len(clean) < 2:
            continue
        letters = [c for c in clean if c.isalpha()]
        if not letters:
            continue
        if sum(1 for c in letters if c.isupper()) / len(letters) >= 0.75:
            libelle_tokens.append(token)
    return " ".join(libelle_tokens)


def _get_ae_cp(words: list, anchor_top: float, y_tol: int = 12) -> tuple[int | None, int | None]:
    def find_words(x_min, x_max):
        return [
            w["text"] for w in words
            if x_min <= w["x0"] <= x_max
            and abs(w["top"] - anchor_top) <= y_tol
            and re.search(r"\d", w["text"])
        ]

    ae_words = find_words(AE_X_MIN, AE_X_MAX)
    cp_words = find_words(CP_X_MIN, CP_X_MAX)

    if not ae_words:
        ae_words = [w["text"] for w in words
                    if AE_X_MIN <= w["x0"] <= AE_X_MAX
                    and anchor_top <= w["top"] <= anchor_top + 22
                    and re.search(r"\d", w["text"])]
    if not cp_words:
        cp_words = [w["text"] for w in words
                    if CP_X_MIN <= w["x0"] <= CP_X_MAX
                    and anchor_top <= w["top"] <= anchor_top + 22
                    and re.search(r"\d", w["text"])]

    return parse_amount(ae_words), parse_amount(cp_words)


def _is_valid_row(libelle: str, ae, cp) -> bool:
    if not libelle and ae is None and cp is None:
        return False
    for bl in LIBELLE_BLACKLIST:
        if bl in libelle:
            return False
    if libelle and len(libelle.replace(" ", "")) < 4:
        return False
    if libelle.upper().startswith("CHAPITRE"):
        return False
    return not (ae is None and cp is None)


# ── Extraction principale ─────────────────────────────────────────────────────
def _extract_budget_lines(pdf_path: str) -> list[dict]:
    all_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[LF 2025] PDF ouvert : {total_pages} pages")
        print(f"[LF 2025] Traitement pages {PAGE_START + 1} → {min(PAGE_END + 1, total_pages)}")

        for page_idx in range(PAGE_START, min(PAGE_END + 1, total_pages)):
            page     = pdf.pages[page_idx]
            raw_text = page.extract_text() or ""
            if "AE" not in raw_text or "CP" not in raw_text:
                continue
            if "(Unité" not in raw_text and "Milliers" not in raw_text:
                continue

            words       = page.extract_words(x_tolerance=3, y_tolerance=3)
            layout_text = page.extract_text(layout=True) or ""
            lines       = layout_text.split("\n")

            anchors = []
            for i, line in enumerate(lines):
                if CHAPITRE_RE.search(line):
                    continue
                m = ANCHOR_RE.match(line)
                if m:
                    anchors.append((i, m.group(2)))
            if not anchors:
                continue

            code_to_top: dict[str, float] = {}
            for w in words:
                if re.match(r"^\d{3}$", w["text"]) and 87 <= w["x0"] <= 127:
                    code_to_top[w["text"]] = w["top"]

            page_count = 0
            for idx, (line_idx, code) in enumerate(anchors):
                start_line     = max(0, line_idx - 3)
                next_anchor_ln = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(lines)
                end_line       = next_anchor_ln - 3

                libelle_parts = []
                for i in range(start_line, min(end_line + 1, len(lines))):
                    if CHAPITRE_RE.search(lines[i]):
                        break
                    frag = _extract_libelle_from_line(lines[i])
                    if frag:
                        libelle_parts.append(frag)
                libelle = clean_ocr(" ".join(libelle_parts))

                anchor_top = code_to_top.get(code)
                ae_val = cp_val = None
                if anchor_top:
                    ae_val, cp_val = _get_ae_cp(words, anchor_top)

                if not _is_valid_row(libelle, ae_val, cp_val):
                    continue

                all_rows.append({
                    "PAGE":    page_idx + 1,
                    "CODE":    code,
                    "LIBELLE": libelle,
                    "AE":      ae_val,
                    "CP":      cp_val,
                })
                page_count += 1

            if page_count:
                print(f"  Page {page_idx + 1:>3} → {page_count:>3} programme(s)")

    return all_rows


# ── Fonction principale exportée ──────────────────────────────────────────────
def extraire_lf2025(pdf_path: str, out_dir: Path = PROC_DIR) -> pd.DataFrame:
    """
    Extrait toutes les lignes budgétaires de la LF 2024-2025.

    Paramètres
    ----------
    pdf_path : chemin absolu vers le PDF MINFI
    out_dir  : dossier de sortie pour Excel et CSV

    Retour
    ------
    DataFrame avec colonnes : PAGE, CODE, LIBELLE, AE, CP
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _extract_budget_lines(pdf_path)
    if not rows:
        print("[LF 2025] ⚠ Aucune ligne extraite.")
        return pd.DataFrame(columns=["PAGE", "CODE", "LIBELLE", "AE", "CP"])

    df = pd.DataFrame(rows, columns=["PAGE", "CODE", "LIBELLE", "AE", "CP"])
    df = df.drop_duplicates(subset=["CODE", "AE", "CP"]).reset_index(drop=True)

    excel_path = out_dir / f"{FNAME_2025_RAW}.xlsx"
    csv_path   = out_dir / f"{FNAME_2025_RAW}.csv"
    export_excel_formate(df, excel_path, sheet_name="Lignes Budgétaires 2025")
    df.to_csv(str(csv_path), index=False, encoding="utf-8-sig", sep=";")

    print(f"\n[LF 2025] ✓ {len(df)} lignes extraites")
    print(f"  Excel → {excel_path}")
    print(f"  CSV   → {csv_path}")
    return df
