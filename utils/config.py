"""Hydra-style config composition with a zero-dependency loader.

The ``configs/`` tree follows the Hydra layout exactly -- config *groups* are
directories (``configs/dataset``, ``configs/model``, ``configs/degradation``),
an experiment is one file in ``configs/experiment/`` with a ``defaults:`` list
picking one entry per group, and command-line overrides use Hydra's
``key.subkey=value`` syntax including ``group=name`` to swap a whole group.

We resolve that ourselves in ~120 lines instead of depending on
``hydra-core`` + ``omegaconf``.  Rationale (see ``docs/design_doc.md``):

* the repo then runs on a bare ``python + torch + numpy`` install, which
  matters when a teammate is debugging on a lab machine or inside the Orbbec
  SDK container;
* Hydra's working-directory rewriting fights with relative dataset roots;
* migrating later is mechanical, because the YAML layout is already Hydra's.

Not supported on purpose: interpolation beyond ``${a.b}``, sweeps, multirun,
plugins.  If we ever need those, install ``hydra-core`` and delete this file --
the YAML does not have to change.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = ["CONFIG_ROOT", "Config", "apply_overrides", "load_config", "load_experiment"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"

_INTERP = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


class Config(dict):
    """A dict with attribute access, so ``cfg.dataset.root`` works.

    Nested dicts are wrapped lazily on access and on construction, so the object
    stays a plain ``dict`` for YAML/JSON serialisation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)
            elif isinstance(v, list):
                self[k] = [Config(x) if isinstance(x, dict) else x for x in v]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"no config key {name!r}; available: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is ...:
                    raise KeyError(f"config key {dotted!r} not found")
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        def unwrap(v):
            if isinstance(v, Config):
                return {k: unwrap(x) for k, x in v.items()}
            if isinstance(v, dict):
                return {k: unwrap(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unwrap(x) for x in v]
            return v

        return unwrap(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"Available in {path.parent}: "
            f"{sorted(p.stem for p in path.parent.glob('*.yaml')) if path.parent.exists() else '<no such dir>'}"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _deep_merge(base: dict, patch: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    """Parse an override right-hand side with YAML rules (so ``null``/``3``/``[1,2]`` work)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict, overrides: Iterable[str], config_root: Path | None = None) -> dict:
    """Apply Hydra-style ``a.b=value`` overrides in order.

    ``group=name`` where ``group`` is a config-group directory reloads that
    whole group from ``configs/<group>/<name>.yaml`` instead of setting a scalar.
    """
    root = config_root or CONFIG_ROOT
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if "." not in key and (root / key).is_dir():
            group_cfg = _read_yaml(root / key / f"{raw.strip()}.yaml")
            group_cfg.setdefault("name", raw.strip())
            out[key] = group_cfg
            continue
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = _parse_value(raw)
    return out


def _resolve_interpolations(cfg: dict) -> dict:
    """Resolve ``${a.b}`` references and ``${env:VAR}`` / ``${env:VAR,default}``."""
    root = Config(cfg)

    def lookup(dotted: str) -> Any:
        if dotted.startswith("env."):
            name = dotted[4:]
            return os.environ.get(name, "")
        return root.get_path(dotted)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            m = _INTERP.fullmatch(node)
            if m:  # whole-string reference keeps the referenced type
                return lookup(m.group(1))
            return _INTERP.sub(lambda mm: str(lookup(mm.group(1))), node)
        return node

    # two passes so a reference to a reference resolves
    resolved = walk(cfg)
    root = Config(resolved)
    return walk(resolved)


def load_config(
    group: str,
    name: str,
    config_root: Path | None = None,
) -> Config:
    """Load a single group entry, e.g. ``load_config("dataset", "nyuv2")``."""
    root = config_root or CONFIG_ROOT
    data = _read_yaml(root / group / f"{name}.yaml")
    data.setdefault("name", name)
    return Config(data)


def load_experiment(
    experiment: str | Path,
    overrides: Sequence[str] = (),
    config_root: Path | None = None,
) -> Config:
    """Compose one experiment config.

    ``experiment`` is either a name in ``configs/experiment/`` or a path to a
    YAML file.  The file's ``defaults:`` list is resolved first (each entry
    ``group: name`` becomes ``cfg[group]``), then the file's own keys are merged
    on top, then ``overrides``.
    """
    root = config_root or CONFIG_ROOT
    path = Path(experiment)
    if not path.suffix:
        path = root / "experiment" / f"{experiment}.yaml"
    raw = _read_yaml(path)

    defaults = raw.pop("defaults", []) or []
    composed: dict[str, Any] = {}
    for entry in defaults:
        if entry in ("_self_", None):
            continue
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(
                f"{path}: each defaults entry must be a single 'group: name' mapping, got {entry!r}"
            )
        ((group, gname),) = entry.items()
        group_cfg = _read_yaml(root / group / f"{gname}.yaml")
        group_cfg.setdefault("name", gname)
        composed[group] = group_cfg

    composed = _deep_merge(composed, raw)
    composed.setdefault("experiment_name", path.stem)
    composed = apply_overrides(composed, overrides, root)
    composed = _resolve_interpolations(composed)
    return Config(composed)
