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
    from src.config import DIR_DB_EVAL
except ImportError:
    print("❌ Error: Could not find config.py in the src directory.")
    sys.exit(1)

def is_port_in_use(port: int = 5000) -> bool:
    """Checks if a local port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_server():
    """Blocking call to start the MLflow server (for CLI usage)."""
    db_path = Path(DIR_DB_EVAL).resolve()
    backend_uri = f"sqlite:///{db_path}"
    
    command = [
        "mlflow", "server",
        "--backend-store-uri", backend_uri,
        "--port", "5000",
        "--host", "127.0.0.1"
    ]
    
    print(f"🚀 Starting MLflow UI...")
    print(f"📍 Database: {db_path}")
    print(f"🌐 URL: http://127.0.0.1:5000")
    
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped.")
    except Exception as e:
        print(f"❌ Failed to start MLflow server: {e}")

def ensure_mlflow_ui():
    """Starts MLflow UI in the background if not already running."""
    if is_port_in_use(5000):
        print("ℹ️  MLflow UI is already running on http://127.0.0.1:5000")
        return

    db_path = Path(DIR_DB_EVAL).resolve()
    backend_uri = f"sqlite:///{db_path}"
    
    # Start as background process
    print("🚀 Launching MLflow UI in background...")
    subprocess.Popen(
        ["mlflow", "server", "--backend-store-uri", backend_uri, "--port", "5000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True if sys.platform == "win32" else False
    )
    # Give it a moment to spin up
    time.sleep(2)
    print("🌐 MLflow UI should now be available at http://127.0.0.1:5000")

if __name__ == "__main__":
    start_server()
