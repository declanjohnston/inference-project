#!/usr/bin/env python3
"""Launch Streamlit dashboard with ngrok tunnel for public access."""
import os
import subprocess
import sys
from pathlib import Path

from pyngrok import ngrok

PORT = 8080
NGROK_URL_FILE = Path("data/ngrok_url.txt")


def save_url(url: str) -> None:
    """Save ngrok URL to file."""
    NGROK_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
    NGROK_URL_FILE.write_text(url)
    print(f"Ngrok URL saved to {NGROK_URL_FILE}")


def main() -> None:
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)
        print("Ngrok auth token set from NGROK_AUTH_TOKEN environment variable")
    
    print(f"Starting ngrok tunnel on port {PORT}...")
    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url

    print(f"\n{'='*60}")
    print(f"Dashboard is publicly accessible at:")
    print(f"  {public_url}")
    print(f"{'='*60}\n")

    save_url(public_url)

    print("Starting Streamlit dashboard...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ngrok.disconnect(tunnel.public_url)
        print("Ngrok tunnel closed.")


if __name__ == "__main__":
    main()
