#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_ref_inputs_no_align.py

Automatically reads inputs from the local reference script (default: ``ref_0.py``)
and benchmarks **only** ``ModelNew`` from the candidate script (default: ``test_kernel_0.py``).
Does **not** run the reference model and performs **no** parameter alignment.
Prints a single JSON line to stdout.
"""
from __future__ import annotations

import os
import sys
import io
import time
import json
import tempfile
import contextlib
import importlib.util
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# Disable cuDNN to avoid being affected by cuDNN operator selection
os.environ["CUDNN_DISABLE"] = "1"
os.environ["PYTORCH_NO_CUDNN"] = "1"
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CompilationError(RuntimeError):
    pass


def _capture_import(path: Path):
    """Dynamically import the given Python file and capture its stdout/stderr.

    Returns
    -------
    (module, full_log: str)

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    CompilationError
        If any error occurs during import/build; the exception message contains
        the concatenated Python + subprocess log.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    py_buf = io.StringIO()
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
            contextlib.redirect_stdout(py_buf), \
            contextlib.redirect_stderr(py_buf):

        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)
            os.dup2(fd_buf.fileno(), 2)
            spec.loader.exec_module(module)  # pyright: ignore[attr-defined]
            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()
        except Exception as exc:
            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    return module, py_buf.getvalue() + subproc_log


def _run_once(model: nn.Module, inp: List[Any], dev: torch.device) -> Tuple[Any, float]:
    """Single forward pass with timing in milliseconds.

    Uses CUDA events on GPU, or wall-clock time on CPU.
    """
    model.to(dev).eval()
    inp = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inp]

    if TORCH_DEVICE == "cpu":
        with torch.inference_mode():
            t0 = time.time()
            out = model(*inp)
            dt_ms = (time.time() - t0) * 1_000.0
        return out, dt_ms

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    with torch.inference_mode():
        start.record()
        out = model(*inp)
        end.record()
        end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: nn.Module, inp: List[Any], dev: torch.device, warm: int, rep: int) -> List[float]:
    """Multiple forward passes; return a list of latencies in ms."""
    model.to(dev).eval()
    inp = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inp]

    # Warmup
    if TORCH_DEVICE == "cpu":
        with torch.inference_mode():
            for _ in range(max(0, warm)):
                model(*inp)
    else:
        with torch.inference_mode():
            for _ in range(max(0, warm)):
                model(*inp)
        torch.cuda.synchronize(dev)

    # Measure
    if TORCH_DEVICE == "cpu":
        res: List[float] = []
        with torch.inference_mode():
            for _ in range(max(1, rep)):
                t0 = time.time()
                model(*inp)
                res.append((time.time() - t0) * 1_000.0)
        return res

    times: List[float] = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        for _ in range(max(1, rep)):
            s.record()
            model(*inp)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
    return times


def _first_tensor(x: Any) -> torch.Tensor:
    """Extract the first Tensor from *x*; raise if none is found."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for t in x:
            if isinstance(t, torch.Tensor):
                return t
    raise TypeError("forward() did not return a Tensor")


def main():
    ap = argparse.ArgumentParser("Bench ModelNew (ref: ref.py, test: test_kernel.py)")
    ap.add_argument("--ref", type=Path, default=Path.cwd() / "ref_0.py")
    ap.add_argument("--test", type=Path, default=Path.cwd() / "test_kernel_0.py")
    ap.add_argument("--device-idx", type=int, default=0)  # safer default value
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    # Absolute paths to avoid CWD differences (e.g., under sudo)
    args.ref = args.ref.resolve()
    args.test = args.test.resolve()

    # Device selection: honor CUDA_VISIBLE_DEVICES; fallback to 0 if out of range
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if TORCH_DEVICE == "cuda":
        num = torch.cuda.device_count()
        idx = args.device_idx if num > 0 else 0
        if idx < 0 or idx >= num:
            print(f"[warn] device-idx {idx} out of range (0..{num-1}), fallback to 0", file=sys.stderr)
            idx = 0
        dev = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")

    print(f"[info] using device: {dev}, CUDA_VISIBLE_DEVICES={visible}", file=sys.stderr)
    print(f"[info] ref={args.ref}, test={args.test}", file=sys.stderr)

    # Dynamic import of ref/test
    ref_mod, _ = _capture_import(args.ref)
    tst_mod, _ = _capture_import(args.test)

    get_inputs = getattr(ref_mod, "get_inputs", None)
    get_init_inputs_ref = getattr(ref_mod, "get_init_inputs", None)
    ModelNew = getattr(tst_mod, "ModelNew", None)
    if get_inputs is None:
        raise RuntimeError(f"{args.ref} must define get_inputs()")
    if ModelNew is None:
        raise RuntimeError(f"{args.test} must define class ModelNew")

    # Init args come from ref's get_init_inputs (if provided)
    init_args, init_kwargs = [], {}
    if callable(get_init_inputs_ref):
        init_obj = get_init_inputs_ref()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)

    # Prepare inputs
    inp = get_inputs()
    if not isinstance(inp, (list, tuple)):
        inp = [inp]

    # Build candidate model
    model = ModelNew(*init_args, **init_kwargs)

    # Run once to obtain output summary
    out, _ = _run_once(model, inp, dev)
    if TORCH_DEVICE == "cuda":
        torch.cuda.synchronize(dev)
    out_t = _first_tensor(out).contiguous()
    print(f"[info] out: shape={list(out_t.shape)}, dtype={out_t.dtype}, device={out_t.device}", file=sys.stderr)

    # Benchmark
    times = _bench(model, inp, dev, args.warmup, args.repeat)
    if TORCH_DEVICE == "cuda":
        torch.cuda.synchronize(dev)
        time.sleep(0.05)  # small cushion for profiler flush/attach

    result: Dict[str, Any] = {
        "reference_file": str(args.ref),
        "candidate_file": str(args.test),
        "device": str(dev),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "latency_ms": {
            "avg": sum(times) / len(times) if times else float("nan"),
            "min": min(times) if times else float("nan"),
            "max": max(times) if times else float("nan"),
            "all": times,
        },
        "output_summary": {
            "shape": list(out_t.shape),
            "dtype": str(out_t.dtype),
            "device": str(out_t.device),
        },
        "cudnn": {
            "env_CUDNN_DISABLE": os.environ.get("CUDNN_DISABLE"),
            "env_PYTORCH_NO_CUDNN": os.environ.get("PYTORCH_NO_CUDNN"),
            "torch_backends_enabled": torch.backends.cudnn.enabled,
            "torch_benchmark": torch.backends.cudnn.benchmark,
        },
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except CompilationError as e:
        sys.stderr.write(str(e) + "\n")
        sys.exit(1)
