# tests/conftest.py
"""Shared pytest fixtures and environment setup."""
import os

# Ensure env vars are set before any cwt_ads_agent imports
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-key")
os.environ.setdefault("LOG_LEVEL", "WARNING")
