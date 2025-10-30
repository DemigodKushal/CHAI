# main.py
import subprocess
import sys
import os

def run_streamlit():
    app_path = os.path.join(os.path.dirname(__file__), "app_ui.py")
    command = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(command)

if __name__ == "__main__":
    run_streamlit()
