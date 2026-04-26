"""Install the CUDA preload bootstrap into the current uv venv.

Writes `sitecustomize.py` into the active venv's site-packages. That
file preloads the pip-bundled NVIDIA shared libraries via
ctypes.RTLD_GLOBAL so JAX can find cuSPARSE, cuBLAS, cuDNN, etc.,
without the user having to set LD_LIBRARY_PATH.

Run once after every `uv sync` (the venv gets rebuilt from scratch
on a sync, so sitecustomize.py is lost).

Usage:
    uv run python scripts/install_gpu_bootstrap.py
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

SITECUSTOMIZE_CONTENT = '''"""Bootstrap CUDA library loading for pip-bundled NVIDIA packages.

Two-stage bootstrap:

1. If ``LD_LIBRARY_PATH`` is missing the wheel-shipped ``nvidia/*/lib``
   directories at its head, prepend them and re-exec the same Python
   process so glibc picks up the corrected search path on its next
   ``_dl_init_paths``. This is the only way to dodge a system CUDA
   toolkit (e.g. ``/usr/local/cuda-12.1``) registered via
   ``/etc/ld.so.conf.d/cuda-*.conf`` or shell env: glibc caches
   ``LD_LIBRARY_PATH`` at process start, so ``os.environ`` updates done
   from sitecustomize.py are too late for any later ``dlopen()``.
   Without this, JAX-plugin SONAME lookups like ``libcublas.so.12`` and
   ``libcusolver.so.11`` resolve to mismatched system libraries and
   crash with errors such as
   ``cusolverDnCreate failed: cuSolver internal error`` or
   ``undefined symbol: cublasSetEnvironmentMode``.

2. Preload each wheel CUDA shared library via
   ``ctypes.CDLL(..., RTLD_GLOBAL)``. Defense in depth and removes the
   need for absolute-path knowledge in JAX's plugin code.

A sentinel env var (``_JAX_GPU_BOOTSTRAP_EXECED``) prevents infinite
re-exec loops. Safe to import on systems without the nvidia extras
(ImportError is caught and the function does nothing).
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys


_PRELOAD_ORDER: tuple[str, ...] = (
    "cuda_runtime",
    "nvjitlink",
    "cuda_cupti",
    "cuda_nvrtc",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cudnn",
    "nccl",
    "nvshmem",
)

_REEXEC_SENTINEL = "_JAX_GPU_BOOTSTRAP_EXECED"


def _wheel_lib_dirs(base: str) -> list[str]:
    return [
        d for d in (os.path.join(base, c, "lib") for c in _PRELOAD_ORDER)
        if os.path.isdir(d)
    ]


def _maybe_reexec_with_wheel_paths() -> None:
    if os.environ.get(_REEXEC_SENTINEL) == "1":
        return
    try:
        import nvidia  # type: ignore[import]
    except ImportError:
        return
    base = (
        os.path.dirname(nvidia.__file__)
        if getattr(nvidia, "__file__", None)
        else list(nvidia.__path__)[0]
    )
    wheel_dirs = _wheel_lib_dirs(base)
    if not wheel_dirs:
        return
    expected_prefix = os.pathsep.join(wheel_dirs)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing.startswith(expected_prefix):
        return
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = (
        expected_prefix + (os.pathsep + existing if existing else "")
    )
    new_env[_REEXEC_SENTINEL] = "1"
    # sys.orig_argv (Python 3.10+) preserves -c CODE, -m MODULE, and -X
    # interpreter flags that sys.argv strips. Required for transparent
    # re-exec of arbitrary Python invocations.
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    os.execvpe(sys.executable, argv, new_env)


def _preload_cuda() -> None:
    try:
        import nvidia  # type: ignore[import]
    except ImportError:
        return

    base = (
        os.path.dirname(nvidia.__file__)
        if getattr(nvidia, "__file__", None)
        else list(nvidia.__path__)[0]
    )
    for component in _PRELOAD_ORDER:
        pattern = os.path.join(base, component, "lib", "*.so*")
        for so_path in sorted(glob.glob(pattern)):
            # Skip unversioned symlinks to avoid loading twice.
            if so_path.endswith(".so"):
                continue
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_maybe_reexec_with_wheel_paths()
_preload_cuda()
'''


def main() -> int:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    if not site_packages.exists():
        print(f"site-packages not found at {site_packages}", file=sys.stderr)
        return 1

    target = site_packages / "sitecustomize.py"
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"Wrote {target}")
    print("JAX should now auto-detect a GPU on import (no LD_LIBRARY_PATH needed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
