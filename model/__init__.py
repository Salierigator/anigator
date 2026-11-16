"""Recommender architectures, one per folder: model/<name>/serve.py exposes load()."""
from importlib import import_module


def load(name: str):
    """The serving recommender of architecture `name`, as a model.base.Recommender."""
    return import_module(f"model.{name}.serve").load()
