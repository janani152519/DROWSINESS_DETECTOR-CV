"""Convenience root launcher for CampusGuard AI Monitor.
Allows running `python run.py` directly from the root repository folder.
"""
import os
import sys
from pathlib import Path

# Change working directory to campus-monitor subfolder
project_dir = Path(__file__).parent / "campus-monitor"
if project_dir.exists():
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

import server

def get_free_port(preferred=5050):
    import socket
    for p in [preferred, 8080, 8001, 8002]:
        try:
            s = socket.socket()
            s.bind(('0.0.0.0', p))
            s.close()
            return p
        except Exception:
            continue
    return preferred

if __name__ == "__main__":
    import uvicorn
    ssl_config = dict(ssl_keyfile="key.pem", ssl_certfile="cert.pem") if Path("key.pem").exists() else {}
    lan_ip = server.get_lan_ip()
    port = get_free_port(server.SERVER_PORT)
    server.SERVER_PORT = port
    print("=" * 65)
    print("🎓 CAMPUSGUARD & ATTENTION-AI FOCUS MONITOR")
    print(f"👉 Local Dashboard:   http://localhost:{port}")
    print(f"👉 Network / Phone:   http://{lan_ip}:{port}")
    print("=" * 65)
    uvicorn.run(server.app, host="0.0.0.0", port=port, **ssl_config)

