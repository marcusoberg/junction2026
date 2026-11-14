import subprocess
import modal

app = modal.App("aalto-adk-gpu-probe")

@app.function(gpu="L40S", timeout=120)
def probe_l40s():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()

@app.local_entrypoint()
def main():
    print("L40S:", probe_l40s.remote())
