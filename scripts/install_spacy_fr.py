# Script d'installation automatique du modèle spaCy français
import subprocess
import sys

def install_spacy_fr_model():
    try:
        import spacy
        spacy.load("fr_core_news_sm")
        print("Le modèle spaCy 'fr_core_news_sm' est déjà installé.")
    except (ImportError, OSError):
        print("Installation du modèle spaCy 'fr_core_news_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
        print("Modèle 'fr_core_news_sm' installé avec succès.")

if __name__ == "__main__":
    install_spacy_fr_model()
