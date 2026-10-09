"""Performance benchmark for the local LLM classifier on *this* machine.

Measures per-headline latency, total batch time, whether a full pipeline batch
(``pipeline.max_classifications_per_run`` headlines) fits inside the 15-minute
interval, memory use of this process and of the Ollama server, and whether Ollama
loaded the model on CPU or GPU (from ``/api/ps``: ``size_vram`` > 0 means GPU).

``--cpu-only`` sends ``num_gpu: 0`` so Ollama keeps every layer on the CPU — use this
to test the "no dedicated GPU" requirement on machines that do have one (Apple
Silicon Macs use the GPU by default).
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from voterpulse.classification.llm_classifier import OllamaClassifier, OllamaUnavailable
from voterpulse.classification.validator import LLMResponseError
from voterpulse.config import AppConfig


def ollama_processes_rss_mb() -> float:
    total = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (proc.info["name"] or "").lower():
                total += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / 2**20, 1)


def loaded_model_placement(client: OllamaClassifier) -> list[dict]:
    try:
        response = client.session.get(f"{client.base_url}/api/ps", timeout=10)
        response.raise_for_status()
        return [{"name": mdl.get("name"), "size_mb": round(mdl.get("size", 0) / 2**20, 1),
                 "size_vram_mb": round(mdl.get("size_vram", 0) / 2**20, 1),
                 "placement": "GPU/unified (partially or fully)" if mdl.get("size_vram", 0) else "CPU only"}
                for mdl in response.json().get("models", [])]
    except Exception as exc:  # reporting only
        return [{"error": str(exc)}]


def run_benchmark(cfg: AppConfig, headlines: list[str], out_dir: Path, cpu_only: bool = False,
                  warmup: int = 1) -> dict:
    client = OllamaClassifier.from_config(cfg)
    if cpu_only:
        client.options["num_gpu"] = 0
    model_id = client.resolve_model_id()
    for text in headlines[:warmup]:  # first call includes model load time; reported separately
        start = time.perf_counter()
        try:
            client.classify(text)
        except (OllamaUnavailable, LLMResponseError):
            pass
        load_s = time.perf_counter() - start
    proc = psutil.Process()
    latencies, errors = [], 0
    peak_ollama = 0.0
    start_all = time.perf_counter()
    for text in headlines:
        try:
            latencies.append(client.classify(text).latency_ms)
        except (OllamaUnavailable, LLMResponseError):
            errors += 1
        peak_ollama = max(peak_ollama, ollama_processes_rss_mb())
    total_s = time.perf_counter() - start_all
    batch = cfg.settings.pipeline.max_classifications_per_run
    mean_ms = statistics.mean(latencies) if latencies else None
    projected = (mean_ms * batch / 1000) if mean_ms else None
    vm = psutil.virtual_memory()
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                    "cpu_count": psutil.cpu_count(logical=True), "total_ram_gb": round(vm.total / 2**30, 2)},
        "model_id": model_id, "cpu_only_requested": cpu_only, "options": client.options,
        "headlines": len(headlines), "errors": errors,
        "first_call_seconds_including_load": round(load_s, 2) if warmup and headlines else None,
        "mean_ms": mean_ms, "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
        "total_seconds": round(total_s, 2),
        "batch_size_per_run": batch,
        "projected_batch_seconds": round(projected, 1) if projected else None,
        "batch_fits_in_interval": (projected < cfg.settings.pipeline.interval_minutes * 60) if projected else None,
        "this_process_rss_mb": round(proc.memory_info().rss / 2**20, 1),
        "ollama_peak_rss_mb": peak_ollama,
        "system_available_ram_gb_after": round(psutil.virtual_memory().available / 2**30, 2),
        "loaded_models": loaded_model_placement(client),
        "note": "Ollama RSS can under-count memory mapped from the model file; also check the OS activity monitor.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
