"""
scripts/run_dashboard.py
========================
Lance le dashboard Streamlit.

Usage
-----
    poetry run python scripts/run_dashboard.py
    # ou
    poetry run streamlit run audit_snd30/dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent.parent / "audit_snd30" / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
