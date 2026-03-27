"""
Convenience script to launch the MLflow UI.
Automatically retrieves the backend store path from config.py.
"""
import subprocess
import sys
import socket
import time
from pathlib import Path

# --- Configuration ---
try:
    from src.config import DIR_DB_EVAL, PROJECT_ROOT
except ImportError:
    print("Error: Could not find config.py in the src directory.")
    sys.exit(1)

# --- Logging ---
from src.utils.m_log import f_log


def is_port_in_use(port: int = 5000) -> bool:
    """Checks if a local port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_server():
    """Blocking call to start the MLflow server (for CLI usage)."""
    rel_db_path = Path(DIR_DB_EVAL).relative_to(PROJECT_ROOT).as_posix()
    backend_uri = f"sqlite:///{rel_db_path}"
    
    command = [
        "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--port", "5000",
        "--host", "127.0.0.1"
    ]
    
    f_log("Starting MLflow UI...", c_type="start")
    f_log(f"Database: {DIR_DB_EVAL}")
    f_log("URL: http://127.0.0.1:5000")
    
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        f_log("MLflow UI stopped.", c_type="warning")
    except Exception as e:
        f_log(f"Failed to start MLflow server: {e}", c_type="error")

def ensure_mlflow_ui():
    """Starts MLflow UI in the background if not already running."""
    if is_port_in_use(5000):
        f_log("MLflow UI is already running on http://127.0.0.1:5000")
        return

    rel_db_path = Path(DIR_DB_EVAL).relative_to(PROJECT_ROOT).as_posix()
    backend_uri = f"sqlite:///{rel_db_path}"
    
    f_log("Launching MLflow UI in background...", c_type="process")
    subprocess.Popen(
        ["mlflow", "server", "--backend-store-uri", backend_uri, "--port", "5000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True if sys.platform == "win32" else False
    )
    time.sleep(2)
    f_log("MLflow UI available at http://127.0.0.1:5000", c_type="success")

if __name__ == "__main__":
    start_server()
