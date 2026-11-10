"""Configuration: YAML file + command-line overrides, with attribute access."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PKG_ROOT / "config" / "default.yaml"


class AttrDict(dict):
    """dict with attribute access, recursively. Also satisfies FlyPose's EasyDict use."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))


def _expand(v):
    if isinstance(v, str) and (v.startswith("~") or "$" in v):
        return os.path.expanduser(os.path.expandvars(v))
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    return v


def _set_dotted(d, dotted, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"unknown config section '{k}' in '{dotted}'")
        cur = cur[k]
    if keys[-1] not in cur:
        raise KeyError(f"unknown config key '{dotted}'")
    cur[keys[-1]] = value


def load_config(path=None, overrides=(), layers=()):
    """Load YAML (default config/default.yaml), apply layer files, then 'a.b=value' overrides.

    Order: default.yaml <- path (user file) <- layers (e.g. config/no_lidar.yaml) <- overrides.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if path != DEFAULT_CONFIG:
        # a user file only needs the keys it changes
        with open(DEFAULT_CONFIG) as f:
            base = yaml.safe_load(f)
        raw = _merge(base, raw)
    for layer in layers or ():
        with open(layer) as f:
            raw = _merge(raw, yaml.safe_load(f) or {})
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        _set_dotted(raw, k.strip(), yaml.safe_load(v))
    return AttrDict(_expand(raw))


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
