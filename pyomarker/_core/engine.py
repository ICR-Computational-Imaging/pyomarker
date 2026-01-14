""" Engine for HMC sampling via cmdstanpy """

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from cmdstanpy import CmdStanModel


@dataclass(frozen=True)
class StanConfig:
    chains: int = 4
    parallel_chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    seed: Optional[int] = None
    adapt_delta: Optional[float] = None
    max_treedepth: Optional[int] = None
    show_progress: bool = True


class StanEngine:
    """ A sampling engine for computing

    - Caches compiled executables based on a hash of Stan source + CmdStan settings.
    - Separates compilation from sampling so models can be stateful and reuse the engine.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".stan_cache",
        *,
        cpp_options: Optional[Dict[str, str]] = None,
        stanc_options: Optional[Dict[str, str]] = None,
        force_recompile: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cpp_options = cpp_options or {}
        self.stanc_options = stanc_options or {}
        self.force_recompile = force_recompile

        # in-memory model cache to avoid re-instantiating CmdStanModel repeatedly
        self._model_cache: Dict[str, CmdStanModel] = {}

    def compile_file(self, stan_file: str | Path) -> CmdStanModel:
        """
        Compile a Stan program from a .stan file and cache the resulting executable.
        Returns a CmdStanModel ready to sample.
        """
        stan_file = Path(stan_file)
        if not stan_file.exists():
            raise FileNotFoundError(f"Stan file not found: {stan_file}")

        code = stan_file.read_text(encoding="utf-8")
        key = self._cache_key(code=code, stan_path=str(stan_file))

        if not self.force_recompile and key in self._model_cache:
            return self._model_cache[key]

        # Place a copy of the stan file in cache under its hash.
        cached_stan = self.cache_dir / f"{key}.stan"
        if not cached_stan.exists() or self.force_recompile:
            cached_stan.write_text(code, encoding="utf-8")

        # CmdStanPy will compile to an executable next to the cached stan file (in cache_dir).
        model = CmdStanModel(
            stan_file=str(cached_stan),
            stanc_options=self.stanc_options or None,
            cpp_options=self.cpp_options or None,
        )

        self._model_cache[key] = model
        return model

    def sample(
        self,
        model: CmdStanModel,
        *,
        data: Dict[str, Any],
        inits: Optional[Dict[str, Any]] = None,
        config: Optional[SampleConfig] = None,
        **kwargs: Any,
    ):
        """
        Run MCMC sampling with CmdStanPy.
        Extra kwargs are passed through to CmdStanModel.sample.
        """
        cfg = config or SampleConfig()

        # CmdStanPy uses None to mean "do not set".
        sampling_kwargs: Dict[str, Any] = dict(
            data=data,
            inits=inits,
            chains=cfg.chains,
            parallel_chains=cfg.parallel_chains,
            iter_warmup=cfg.iter_warmup,
            iter_sampling=cfg.iter_sampling,
            seed=cfg.seed,
            show_progress=cfg.show_progress,
        )

        # Adaptation controls
        if cfg.adapt_delta is not None:
            sampling_kwargs["adapt_delta"] = cfg.adapt_delta
        if cfg.max_treedepth is not None:
            sampling_kwargs["max_treedepth"] = cfg.max_treedepth

        # Allow caller to override anything
        sampling_kwargs.update(kwargs)

        return model.sample(**sampling_kwargs)

    def clear_memory_cache(self) -> None:
        """Clear only the in-memory cache (does not delete compiled artifacts on disk)."""
        self._model_cache.clear()

    # -----------------------
    # internal helpers
    # -----------------------
    def _cache_key(self, *, code: str, stan_path: str) -> str:
        """
        Hash the Stan code + relevant compile options to create a stable cache key.
        """
        h = hashlib.sha256()
        h.update(code.encode("utf-8"))
        h.update(b"\0")
        h.update(os.path.abspath(stan_path).encode("utf-8"))
        h.update(b"\0")
        h.update(repr(sorted(self.cpp_options.items())).encode("utf-8"))
        h.update(b"\0")
        h.update(repr(sorted(self.stanc_options.items())).encode("utf-8"))
        return h.hexdigest()[:16]
