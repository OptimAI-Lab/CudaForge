from __future__ import annotations
"""
compare_and_bench.py – single-GPU benchmark (**full compile + runtime traceback**).

Key features
------------
* Dynamically imports two PyTorch models (reference & candidate) and **captures
  every byte** printed by Python *and* child processes (ninja / nvcc).
  - On any *build* failure, raises `CompilationError(full_log)`.
* On **runtime failure** (forward, benchmark, accuracy), re-raises
  `RuntimeError(traceback.format_exc())` so callers get the *entire*
  traceback – not just `str(exc)`.
* Benchmarks on CUDA (default) or CPU (`--cpu`).
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails.

    The *first* argument is the full build log (Python + ninja/nvcc).
    """


class AccuracyError(RuntimeError):
    """Raised when outputs do not meet the accuracy tolerance."""


# =========================== dynamic import ===============================
def _capture_import(path: Path):
    """Import *path* dynamically and capture **all** build logs.

    Returns
    -------
    (module, full_log : str)

    Raises
    ------
    FileNotFoundError
        *path* does not exist.
    CompilationError
        Any Python / ninja / nvcc error during import.  The exception's first
        argument is the concatenated log.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Python-level stdout/stderr to StringIO --------------------------
    py_buf = io.StringIO()

    # ---- OS-level FD 1/2 (stdout/stderr) to a temp file -----------------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # Save current FDs so we can restore later
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ REAL IMPORT (build/compile) --------------------
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except Exception as exc:  # ← build / link / import failed
            # Combine StringIO + temp-file logs + Exception str
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            raise CompilationError(full_log) from None

        finally:
            # Always restore original FDs
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- SUCCESS --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


# =========================== timing helpers ===============================
def _run_once(model: torch.nn.Module,
              inp: List[torch.Tensor],
              dev: torch.device) -> Tuple[torch.Tensor, float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    if TORCH_DEVICE == "cpu":
        t0 = datetime.now()
        out = model(*inp)
        ms = (datetime.now() - t0).total_seconds() * 1_000
        return out, ms

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    start.record()
    out = model(*inp)
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: torch.nn.Module,
           inp: List[torch.Tensor],
           dev: torch.device,
           warm: int,
           rep: int) -> List[float]:
    model.to(dev).eval()
    inp = [x.to(dev) for x in inp]

    for _ in range(warm):
        model(*inp)

    if TORCH_DEVICE == "cpu":
        res = []
        for _ in range(rep):
            t0 = datetime.now()
            model(*inp)
            res.append((datetime.now() - t0).total_seconds() * 1_000)
        return res

    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times: List[float] = []
    for _ in range(rep):
        s.record()
        model(*inp)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return times


# ====================== RNG & 确定性设置 ======================
def _seed_everything(seed: int | None, device_idx: int | None = None):
    """设置随机种子并（可选）启用确定性后端。"""
    import os, random
    import numpy as np
    import torch

    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device_idx is not None:
            torch.cuda.set_device(device_idx)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 更强可复现（如不需要可注释掉）
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 或 ":16:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 某些算子无确定性实现时仅告警不报错
        torch.use_deterministic_algorithms(True, warn_only=True)


# ====================== 参数对齐（通用 + 类名/导出名专用） ======================
import torch
import torch.nn as nn
from collections import defaultdict

def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named

@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True

@torch.no_grad()
def _try_map_shape_and_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    形状映射覆盖：
      - depthwise 2D:   (C,1,Kh,1)<->(C,Kh), (C,1,Kh,Kw)<->(C,Kh,Kw)
      - PW/Linear:      (Out,In,1,1)<->(Out,In)
      - Conv/ConvT 3D:  (Out,In,kD,kH,kW) <-> (In,Out,kD,kH,kW) （首两维交换）
      - depthwise 3D:   (C,1,kD,kH,kW) <-> (C,kD,kH,kW)
    """
    s = tuple(src.shape)
    d = tuple(dst.shape)

    # --- depthwise 2D: (C,1,Kh,1) <-> (C,Kh)
    if len(s) == 4 and s[1] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[2] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[1] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- depthwise 2D: (C,1,Kh,Kw) -> (C,Kh,Kw) 及其反向
    if len(s) == 4 and s[1] == 1 and len(d) == 3 and s[0] == d[0] and s[2] == d[1] and s[3] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 3 and len(d) == 4 and d[1] == 1 and s[0] == d[0] and s[1] == d[2] and s[2] == d[3]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    # --- PW/Linear: (Out,In,1,1) <-> (Out,In)
    if len(s) == 4 and s[2] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[2] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- Conv/ConvTranspose 3D: 5D 权重首两维交换
    #     (Out, In, kD, kH, kW)  <->  (In, Out, kD, kH, kW)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # --- depthwise 3D: (C,1,kD,kH,kW) -> (C,kD,kH,kW) 及其反向
    if len(s) == 5 and s[1] == 1 and len(d) == 4 and s[0] == d[0] and s[2:] == d[1:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 4 and len(d) == 5 and d[1] == 1 and s[0] == d[0] and s[1:] == d[2:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    return False

@torch.no_grad()
def align_params_generic(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, mapped, skipped = 0, 0, 0, 0
    aligned_test: set[str] = set()

    # 1) 同名同形状
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) 唯一形状匹配
    shape2ref: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    shape2test: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    for n, t in test_named.items():
        if n in aligned_test: 
            continue
        shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                unique_shape_copied += 1
                aligned_test.add(tname)

    # 3) 形状映射
    for name, t_dst in test_named.items():
        if name in aligned_test:
            continue
        ok = False
        for _, t_src in ref_named.items():
            if _try_map_shape_and_copy_(t_dst, t_src):
                mapped += 1
                aligned_test.add(name)
                ok = True
                break
        if not ok:
            skipped += 1

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "mapped_shape": mapped,
        "skipped": skipped,
    }

# ——（可选）按类名/导出名注册“专用对齐器”：Model → ModelNew ——
_PAIR_ALIGNERS: dict[tuple[str, str], callable] = {}

def register_pair_aligner(ref_key: str, test_key: str):
    def deco(fn):
        _PAIR_ALIGNERS[(ref_key, test_key)] = fn
        return fn
    return deco

@register_pair_aligner("Model", "ModelNew")
@torch.no_grad()
def _align_Model_to_ModelNew(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    def pick(named: dict[str, torch.Tensor], dims: int):
        cand = [(n, t) for n, t in named.items()
                if n.startswith("param::") and "weight" in n and t.ndim == dims]
        if not cand:
            cand = [(n, t) for n, t in named.items()
                    if n.startswith("param::") and t.ndim == dims]
        return cand

    # ---- 2D: Conv / ConvTranspose（4D 同形状 或 首两维交换）----
    r4 = pick(ref_named, 4); t4 = pick(test_named, 4)
    if len(r4) == 1 and len(t4) == 1:
        w_ref, w_tst = r4[0][1], t4[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        elif (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
              and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            pass_bias = True
        else:
            pass_bias = False

        if pass_bias:
            rb = [(n,t) for n,t in ref_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            tb = [(n,t) for n,t in test_named.items() if "bias" in n and n.startswith("param::") and t.ndim==1]
            if len(rb)==1 and len(tb)==1 and tuple(rb[0][1].shape)==tuple(tb[0][1].shape):
                tb[0][1].copy_(rb[0][1].to(dtype=tb[0][1].dtype, device=tb[0][1].device))
            return {"pair_aligner": 1, "copied_same_shape": int(tuple(w_ref.shape)==tuple(w_tst.shape)),
                    "mapped_shape": int(tuple(w_ref.shape)!=tuple(w_tst.shape)), "skipped": 0}

    # ---- 3D: Conv3d / ConvTranspose3d（5D 同形状 或 首两维交换）----
    r5 = pick(ref_named, 5); t5 = pick(test_named, 5)
    if len(r5) == 1 and len(t5) == 1:
        w_ref, w_tst = r5[0][1], t5[0][1]
        if tuple(w_ref.shape) == tuple(w_tst.shape):
            w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 1, "mapped_shape": 0, "skipped": 0}
        if (w_ref.shape[0] == w_tst.shape[1] and w_ref.shape[1] == w_tst.shape[0]
                and w_ref.shape[2:] == w_tst.shape[2:]):
            w_tst.copy_(w_ref.permute(1, 0, 2, 3, 4).contiguous().to(dtype=w_tst.dtype, device=w_tst.device))
            return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # ---- depthwise-3D: (C,1,kD,kH,kW) ↔ (C,kD,kH,kW) ----
    if len(r5) == 1:
        w_ref = r5[0][1]
        t4 = pick(test_named, 4)
        if len(t4) == 1:
            w_tst = t4[0][1]
            if w_ref.size(1) == 1 and tuple(w_tst.shape) == (w_ref.size(0), w_ref.size(2), w_ref.size(3), w_ref.size(4)):
                w_tst.copy_(w_ref.to(dtype=w_tst.dtype, device=w_tst.device).squeeze(1).contiguous())
                return {"pair_aligner": 1, "copied_same_shape": 0, "mapped_shape": 1, "skipped": 0}

    # 其余回退通用
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    return stats

@torch.no_grad()
def try_align_params(ref_model: nn.Module, test_model: nn.Module,
                     ref_mod=None, test_mod=None) -> dict[str, int]:
    """
    优先级：
      0) 导出名派发（_export_symbol），如 ("Model","ModelNew")
      0b) 实例类名派发
      1) 任务自定义 map_ref_to_test_params / align_params
      2) 通用自动对齐
    """
    # 0) 导出名（若 compare_and_bench 已设置）
    key_export = (getattr(ref_model, "_export_symbol", None),
                  getattr(test_model, "_export_symbol", None))
    if key_export in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_export](ref_model, test_model)
        stats["pair_key"] = f"{key_export[0]}->{key_export[1]}"
        return stats

    # 0b) 实例类名
    key_class = (ref_model.__class__.__name__, test_model.__class__.__name__)
    if key_class in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_class](ref_model, test_model)
        stats["pair_key"] = f"{key_class[0]}->{key_class[1]}"
        return stats

    # 1) 任务自定义
    for mod in (test_mod, ref_mod):
        if mod is None:
            continue
        for fn_name in ("map_ref_to_test_params", "align_params"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(ref_model, test_model)
                return {"pair_aligner": 0, "copied_same_shape": -1, "mapped_shape": -1,
                        "skipped": -1, "pair_key": "custom_fn"}

    # 2) 通用
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    stats["pair_key"] = "generic"
    return stats



# ====================== compare_and_bench（带通用对齐与种子） ======================
def compare_and_bench(
    ref_py: Path,
    test_py: Path,
    *,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 1e-4,
    log_dir: str | Path | None = "run/debug",
    seed: int = 100,  # 固定默认 seed；需要环境控制时可改成 None 并用 env 读取
) -> Dict[str, Any]:
    """
    Benchmark *test_py* against *ref_py*.

    仅从 reference 脚本读取 get_init_inputs()，并对 ref/test 使用同一组初始化参数。
    同时：固定随机性 + 参数对齐（支持 Model→ModelNew 专用对齐 & 通用对齐）。
    """
    import os
    import contextlib
    from datetime import datetime

    # ------------ 设备设置 ------------
    dev = torch.device(f"cuda:{device_idx}") if TORCH_DEVICE == "cuda" else torch.device("cpu")
    if TORCH_DEVICE == "cuda":
        torch.cuda.set_device(dev)

    # 若需要通过环境变量控制 seed
    if seed is None:
        env_seed = os.environ.get("KERNELBENCH_SEED")
        seed = int(env_seed) if env_seed is not None else None

    # ------------ 动态导入 ------------
    ref_mod, _ = _capture_import(ref_py)
    test_mod, _ = _capture_import(test_py)

    RefModel   = getattr(ref_mod,  "Model",       None)
    get_inputs = getattr(ref_mod,  "get_inputs",  None)
    ModelNew   = getattr(test_mod, "ModelNew",    None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{ref_py}' 必须定义 Model 与 get_inputs()。")
    if ModelNew is None:
        raise RuntimeError(f"Candidate '{test_py}' 必须定义 ModelNew 类。")

    # ------------ 仅从 ref 获取初始化参数 ------------
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}
    get_init_inputs_ref = getattr(ref_mod, "get_init_inputs", None)

    if callable(get_init_inputs_ref):
        init_obj = get_init_inputs_ref()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            raise TypeError("get_init_inputs() 必须返回 list/tuple（作为 *args）或 dict（作为 **kwargs）。")

    # ------------ 运行 & 基准 ------------
    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
        raise TypeError("Model forward 未返回 Tensor（或序列中的 Tensor）。")

    try:
        ctx = torch.cuda.device(dev) if TORCH_DEVICE == "cuda" else contextlib.nullcontext()
        with ctx:
            # 固定输入随机性
            _seed_everything(seed, device_idx)
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]

            # 固定参数初始化：两边构造前分别设种子
            _seed_everything(seed, device_idx)
            ref_model  = RefModel(*init_args, **init_kwargs)
            # # torch.compile 加速
            # ref_model = torch.compile(ref_model)
            
            _seed_everything(seed, device_idx)
            test_model = ModelNew(*init_args, **init_kwargs)
            # # torch.compile 加速
            # test_model = torch.compile(test_model)   
            
            # ★ 参数对齐（优先 Model→ModelNew 专用对齐，其次任务自定义，最后通用对齐）
            align_stats = try_align_params(ref_model, test_model, ref_mod=ref_mod, test_mod=test_mod)

            # 前向（同步确保错误就地暴露）
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)
            ref_out,  _ = _run_once(ref_model,  inp, dev)
            test_out, _ = _run_once(test_model, inp, dev)
            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

            # 统一取 Tensor、保证连续
            ref_out  = _first_tensor(ref_out).contiguous()
            test_out = _first_tensor(test_out).contiguous()
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            # Check memory usage
            ref_out_bytes = ref_out.element_size() * ref_out.nelement()

            if ref_out_bytes * 8 > 40 * 1024**3:
                import psutil
                from utils.print_utils import print_warning
                
                # Estimate CPU memory needed (3x safety factor for copy + diff)
                needed_cpu_mem = ref_out_bytes * 3
                avail_cpu_mem = psutil.virtual_memory().available
                
                if avail_cpu_mem < needed_cpu_mem:
                    print_warning(f"Skipping precision check: Tensor too large for both GPU and CPU RAM (Need ~{needed_cpu_mem/1024**3:.1f}GB, Avail {avail_cpu_mem/1024**3:.1f}GB)")
                    check_precision = False
                    # Release tensors to free memory for benchmarking
                    del ref_out, test_out
                    if TORCH_DEVICE == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print_warning(f"Warning: Output tensor size is too large ({ref_out_bytes / 1024**3:.2f} GB). Moving to CPU for comparison to avoid OOM.")
                    ref_out = ref_out.cpu()
                    test_out = test_out.cpu()


            # 误差 & allclose
            diff = (test_out - ref_out).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_out, test_out, atol=tol, rtol=tol):
                raise ValueError(
                    f"Outputs are not close (atol={tol}, rtol={tol}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}"
                )

            # 计时
            ref_t  = _bench(ref_model,  inp, dev, warmup, repeat)
            test_t = _bench(test_model, inp, dev, warmup, repeat)

            if TORCH_DEVICE == "cuda":
                torch.cuda.synchronize(dev)

    except Exception:
        # 抛出完整 traceback（上层捕获）
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None

    # ------------ 结果汇总 ------------
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_file": str(ref_py),
        "candidate_file": str(test_py),
        "tolerance": tol,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "ref_latency_ms": {
            "avg": sum(ref_t) / len(ref_t),
            "min": min(ref_t),
            "max": max(ref_t),
            "all": ref_t,
        },
        "test_latency_ms": {
            "avg": sum(test_t) / len(test_t),
            "min": min(test_t),
            "max": max(test_t),
            "all": test_t,
        },
        "num_runs": repeat,
        "model_init_args": init_args,
        "model_init_kwargs": init_kwargs,
        "seed": seed,
        "align_stats": align_stats,  # 记录对齐信息（含是否命中 Model→ModelNew 专用对齐）
    }
    return result





# =========================== CLI wrapper ==================================
def _cli():
    p = argparse.ArgumentParser(description="Compare & bench two model files.")
    p.add_argument("reference", type=Path, help="Path to reference .py")
    p.add_argument("candidate", type=Path, help="Path to candidate .py")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Benchmark runs")
    p.add_argument("--tol", type=float, default=1e-4, help="Max abs error tolerance")
    p.add_argument("--dump", type=Path, help="If set, write JSON results here")
    args = p.parse_args()

    res = compare_and_bench(
        args.reference,
        args.candidate,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
    )
    print(json.dumps(res, indent=2))

    if args.dump:
        args.dump.write_text(json.dumps(res, indent=2))
        print(f"\nSaved ⇒ {args.dump}")


if __name__ == "__main__":
    _cli()

