"""checkpoint: best.pt from older code layouts and newer Pythons loads into the current config."""
import pathlib
import sys
import types

import torch

from model.retriever import checkpoint, config


def test_load_resolves_legacy_module_names(tmp_path, monkeypatch):
    """Pickle cfg the way old checkpoints name its classes (flat `config`, Python >= 3.13
    `pathlib._local`), then load it in a process that has neither module name registered."""
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.setitem(sys.modules, "pathlib._local", pathlib)
    monkeypatch.setattr(config.TwoTowerConfig, "__module__", "config")
    monkeypatch.setattr(pathlib.PosixPath, "__module__", "pathlib._local")
    torch.save({"cfg": config.TwoTowerConfig(ckpt_dir=pathlib.PosixPath("/content/run"))},
               tmp_path / "best.pt")
    monkeypatch.undo()
    monkeypatch.delitem(sys.modules, "config", raising=False)
    if sys.version_info < (3, 13):
        monkeypatch.delitem(sys.modules, "pathlib._local", raising=False)

    cfg = checkpoint.load(tmp_path / "best.pt")["cfg"]
    assert type(cfg) is config.TwoTowerConfig
    assert cfg.ckpt_dir == pathlib.Path("/content/run")


def test_saved_config_fills_fields_added_later():
    cfg = checkpoint.saved_config(types.SimpleNamespace(d=64, tau=0.05))
    assert (cfg.d, cfg.tau) == (64, 0.05)
    assert cfg.optimizer == config.TwoTowerConfig().optimizer
