"""AI-backed helpers (narrative generation, bill ingestion, tariff Q&A).

Every outbound call is routed through :mod:`modules.ai.client` so that
prompt caching, model selection, and auth are configured in exactly one
place.
"""
