import subprocess
import time
import requests
import logging
import os
import atexit
from config import LLAMA_MODEL_PATH, LLAMA_SERVER_PORT

logger = logging.getLogger(__name__)

class LlamaServerManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaServerManager, cls).__new__(cls)
            cls._instance.process = None
        return cls._instance

    def start(self, model_path=LLAMA_MODEL_PATH, port=LLAMA_SERVER_PORT):
        if self.process is not None:
            logger.info("llama-server is already running.")
            return

        llama_exe = os.environ.get("LLAMA_SERVER_PATH")
        if not llama_exe:
            # Search in the backend directory robustly
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for root, dirs, files in os.walk(backend_dir):
                if "llama-server.exe" in files:
                    llama_exe = os.path.join(root, "llama-server.exe")
                    break
            if not llama_exe and os.path.exists("llama-server.exe"):
                llama_exe = "llama-server.exe"
            if not llama_exe:
                llama_exe = "llama-server"
        
        # Ensure model_path is absolute based on backend directory if it's relative
        if not os.path.isabs(model_path):
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(backend_dir, model_path)
            
        # Detect optimal physical cores for CPU threading
        cpu_count = os.cpu_count() or 8
        threads = max(1, cpu_count // 2)
        
        logger.info(f"Starting llama-server with model {model_path} on port {port} (Threads: {threads})...")
        
        try:
            # Redirect output to a log file instead of DEVNULL to debug crashes
            self.log_file = open("llama_log.txt", "w")
            self.process = subprocess.Popen(
                [
                    llama_exe, 
                    "-m", model_path, 
                    "-c", "8192", 
                    "-ngl", "99", 
                    "-t", str(threads),
                    "--port", str(port)
                ],
                stdout=self.log_file,
                stderr=subprocess.STDOUT
            )
        except FileNotFoundError:
            logger.error(f"Could not find '{llama_exe}'. Please ensure it is installed and in your PATH.")
            return

        # Wait for the server to be healthy and model fully loaded
        max_retries = 120
        for i in range(max_retries):
            try:
                res = requests.get(f"http://localhost:{port}/health", timeout=1)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("status") == "ok":
                        logger.info("llama-server is fully loaded and ready to accept requests.")
                        return
                    else:
                        # Sometimes status is "loading model"
                        pass
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            
        logger.error("llama-server failed to become fully loaded within the timeout period.")

    def stop(self):
        if self.process is not None:
            logger.info("Stopping llama-server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("llama-server stopped.")

# Đảm bảo tắt server khi script exit đột ngột
@atexit.register
def cleanup():
    LlamaServerManager().stop()
