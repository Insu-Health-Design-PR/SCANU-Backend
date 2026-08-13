"""CPU, GPU, and RAM metrics."""

# TODO: psutil / nvidia-smi integration Will do it later !


def collect_system_metrics() -> dict:
    return {"cpu_percent": 0.0, "ram_percent": 0.0, "gpu_percent": None}
