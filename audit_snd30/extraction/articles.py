"""
Module d'extraction et de nettoyage des articles d'une Loi de Finances PDF.
"""
import re
import pdfplumber
import spacy
from pathlib import Path
from typing import List, Dict

# Charger le modèle spaCy français (à installer : python -m spacy download fr_core_news_sm)
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

def nettoyage_complet_loi(texte_brut: str) -> str:
    """Nettoie un texte brut : supprime stopwords, ponctuation, espaces, mots courts."""
    texte = texte_brut.replace('\n', ' ')
    texte = re.sub(r'\s+', ' ', texte)
    doc = nlp(texte)
    tokens_nettoyes = [
        token.text.lower() for token in doc
        if (not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 2)
    ]
    return " ".join(tokens_nettoyes)

def extraire_articles_pdf(pdf_path: Path) -> List[Dict[str, str]]:
    """Extrait et découpe les articles d'un PDF de Loi de Finances."""
    articles_extraits = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        texte_complet = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                texte_complet += page_text + "\n"
    motif = r"(ARTICLE\s+[A-ZÇÉÈÊËÀÂÎÏÔÛÙ-]+(?:\s+[A-ZÇÉÈÊËÀÂÎÏÔÛÙ-]+)*|ARTICLE\s+\d+)"
    segments = re.split(motif, texte_complet)
    for i in range(1, len(segments), 2):
        articles_extraits.append({
            "titre": segments[i].strip(),
            "contenu": segments[i+1].strip() if i+1 < len(segments) else ""
        })
    return articles_extraits

def articles_nettoyes(pdf_path: Path) -> List[Dict[str, str]]:
    """Retourne une liste d'articles nettoyés à partir d'un PDF."""
    articles = extraire_articles_pdf(pdf_path)
    articles_propres = []
    for art in articles:
        titre = art['titre']
        contenu_sale = art['contenu']
        propre = nettoyage_complet_loi(contenu_sale)
        articles_propres.append({
            "article": titre,
            "texte_nettoye": propre
        })
    return articles_propres

# Exemple d'utilisation :
# articles = articles_nettoyes(Path("data/raw/LOI DES FINANCES 2023-2024.pdf"))
# print(articles[:3])
