"""Read Two-Tower checkpoints (best.pt) and the TwoTowerConfig pickled inside them, whatever code
layout or Python version wrote them."""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import torch

from model.twotower.retriever import config


def load(path: pathlib.Path) -> dict:
    """Load a checkpoint onto the CPU."""
    # Checkpoints from before the package layout pickle cfg under the module name `config`;
    # Python >= 3.13 pickles paths as `pathlib._local.PosixPath`, which older Pythons lack.
    sys.modules.setdefault("config", config)
    sys.modules.setdefault("pathlib._local", pathlib)
    return torch.load(path, map_location="cpu", weights_only=False)


def saved_config(saved) -> config.TwoTowerConfig:
    """Rebuild a pickled config with the current class; fields it predates get their defaults."""
    base = config.TwoTowerConfig()
    return config.TwoTowerConfig(**{f.name: getattr(saved, f.name, getattr(base, f.name))
                                    for f in dataclasses.fields(config.TwoTowerConfig)})
