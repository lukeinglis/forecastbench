"""Tests for multi-model routing forecaster."""

from __future__ import annotations

import json


def test_model_for_source_routing():
    from multi_model_forecaster import _model_for_source

    assert _model_for_source("acled") == "vertex_ai/claude-sonnet-4@20250514"
    assert _model_for_source("wikipedia") == "openai/o3-mini"
    assert _model_for_source("dbnomics") == "openai/gpt-4o"
    assert _model_for_source("fred") == "openai/gpt-5-mini"
    assert _model_for_source("yfinance") == "openai/o3-mini"
    assert _model_for_source("manifold") == "openai/gpt-5-mini"
    assert _model_for_source("polymarket") == "openai/gpt-5-mini"
    assert _model_for_source("metaculus") == "vertex_ai/claude-sonnet-4@20250514"
    assert _model_for_source("infer") == "vertex_ai/claude-sonnet-4@20250514"


def test_model_for_source_case_insensitive():
    from multi_model_forecaster import _model_for_source

    assert _model_for_source("ACLED") == "vertex_ai/claude-sonnet-4@20250514"
    assert _model_for_source("Fred") == "openai/gpt-5-mini"
    assert _model_for_source("Wikipedia") == "openai/o3-mini"


def test_model_for_source_default():
    from multi_model_forecaster import _model_for_source, DEFAULT_MODEL

    assert _model_for_source("unknown_source") == DEFAULT_MODEL
    assert _model_for_source("") == DEFAULT_MODEL
    assert _model_for_source("nonexistent") == DEFAULT_MODEL


def test_model_override_env(monkeypatch):
    custom_routing = {"acled": "openai/gpt-4o", "fred": "openai/o3-mini"}
    monkeypatch.setenv("FORECAST_MODEL_ROUTING", json.dumps(custom_routing))

    import importlib
    import multi_model_forecaster
    importlib.reload(multi_model_forecaster)

    assert multi_model_forecaster._model_for_source("acled") == "openai/gpt-4o"
    assert multi_model_forecaster._model_for_source("fred") == "openai/o3-mini"
    assert multi_model_forecaster._model_for_source("wikipedia") == multi_model_forecaster.DEFAULT_MODEL

    monkeypatch.delenv("FORECAST_MODEL_ROUTING")
    importlib.reload(multi_model_forecaster)


def test_default_routing_completeness():
    from multi_model_forecaster import DEFAULT_ROUTING

    expected_sources = {"acled", "wikipedia", "dbnomics", "fred", "yfinance", "manifold", "polymarket", "metaculus", "infer"}
    assert set(DEFAULT_ROUTING.keys()) == expected_sources
