"""Optional experiment tracking with JSONL-first failure behavior."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict


def _normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    if mode == "online":
        return "cloud"
    return mode


def _flatten_numeric(prefix: str, value: Any, output: Dict[str, float | int | bool]) -> None:
    if isinstance(value, bool):
        output[prefix] = value
    elif isinstance(value, (int, float)):
        output[prefix] = value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_key, item, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            next_key = f"{prefix}.{index}" if prefix else str(index)
            _flatten_numeric(next_key, item, output)


def _swanlab_payload(metrics: Mapping[str, Any]) -> Dict[str, float | int | bool]:
    payload: Dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        _flatten_numeric(str(key), value, payload)
    return payload


class ExperimentTracker:
    """Small SwanLab wrapper that never makes training depend on SwanLab."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str | None,
        mode: str,
        logdir: str | Path | None,
        config: Mapping[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.active = False
        self._logger = logger
        self._swanlab = None
        if not enabled or mode == "disabled":
            return

        try:
            import swanlab  # type: ignore

            self._swanlab = swanlab
            swanlab.init(
                project=project,
                experiment_name=run_name,
                config=dict(config),
                logdir=str(logdir) if logdir is not None else None,
                mode=_normalize_mode(mode),
            )
            self.active = True
            self._logger.info("swanlab_active=true project=%s run_name=%s mode=%s", project, run_name, mode)
        except Exception as exc:  # pragma: no cover - depends on optional service state
            self._logger.warning("swanlab_unavailable=true reason=%s; continuing with local JSONL logging", exc)
            self.active = False
            self._swanlab = None

    def log(self, metrics: Mapping[str, Any]) -> None:
        if not self.active or self._swanlab is None:
            return
        payload = _swanlab_payload(metrics)
        if not payload:
            return
        try:
            self._swanlab.log(payload)
        except Exception as exc:  # pragma: no cover - depends on optional service state
            self._logger.warning("swanlab_log_failed=true reason=%s; disabling SwanLab for this run", exc)
            self.active = False

    def finish(self) -> None:
        if not self.active or self._swanlab is None:
            return
        try:
            self._swanlab.finish()
        except Exception as exc:  # pragma: no cover - depends on optional service state
            self._logger.warning("swanlab_finish_failed=true reason=%s", exc)
        finally:
            self.active = False
