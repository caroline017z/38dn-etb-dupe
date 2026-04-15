"""Contract tests for the AI wrappers.

All outbound Anthropic calls are mocked — these tests pin the *shape* of
requests (model, cache_control, tool schema) and the *parsing* of
responses, without touching the live API.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _text_message(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    )


def _tool_message(tool_name: str, tool_input: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name=tool_name, input=tool_input)],
    )


# ---------------------------------------------------------------------------
# client.call_with_cache
# ---------------------------------------------------------------------------
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_call_with_cache_sets_cache_control_on_system_and_context():
    from modules.ai import client as ai_client

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_message("ok")

    ai_client.call_with_cache(
        system="SYSTEM",
        user="USER",
        cached_context=[ai_client.CachedBlock(kind="text", content="CTX")],
        client=mock_client,
    )

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == ai_client.DEFAULT_MODEL
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_content = kwargs["messages"][0]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "CTX"
    assert user_content[0]["cache_control"] == {"type": "ephemeral"}
    assert user_content[-1]["text"] == "USER"
    assert "cache_control" not in user_content[-1]


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_call_with_cache_includes_tools_when_given():
    from modules.ai import client as ai_client

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _text_message("ok")

    tool = {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    ai_client.call_with_cache(
        system="S", user="U", tools=[tool], client=mock_client,
    )
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["tools"] == [tool]


def test_text_from_skips_tool_use_blocks():
    from modules.ai.client import text_from

    msg = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="hello "),
        SimpleNamespace(type="tool_use", name="x", input={}),
        SimpleNamespace(type="text", text="world"),
    ])
    assert text_from(msg) == "hello world"


def test_tool_use_input_returns_matching_block_only():
    from modules.ai.client import tool_use_input

    msg = SimpleNamespace(content=[
        SimpleNamespace(type="tool_use", name="wrong", input={"a": 1}),
        SimpleNamespace(type="tool_use", name="want", input={"a": 2}),
    ])
    assert tool_use_input(msg, "want") == {"a": 2}
    assert tool_use_input(msg, "missing") is None


# ---------------------------------------------------------------------------
# proposal_narrative
# ---------------------------------------------------------------------------
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_proposal_narrative_parses_bullets():
    from modules.ai import proposal_narrative as pn

    ctx = pn.ProposalContext(
        customer_name="Test Farm", site_address="123 Road, Fresno CA",
        system_size_kw=500.0, battery_capacity_kwh=250.0,
        nem_regime="NEM-3", year1_savings_usd=45000.0,
        year1_bill_without_solar_usd=120000.0,
        year1_bill_with_solar_usd=75000.0, savings_pct=37.5,
        horizon_years=20, total_projected_savings_usd=1_200_000.0,
        ppa_rate_usd_per_kwh=0.135,
    )

    with patch.object(pn, "call_with_cache", return_value=_text_message(
        "- Year-1 savings $45,000 (37.5%).\n"
        "- 20-year savings $1.2M.\n"
        "- 500 kW PV with 250 kWh battery under NEM-3.\n"
        "- PPA rate $0.135/kWh."
    )):
        bullets = pn.generate_executive_summary(ctx)

    assert 3 <= len(bullets) <= 5
    assert all(not b.startswith("-") for b in bullets)
    assert any("$45,000" in b for b in bullets)


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_proposal_narrative_falls_back_on_short_response():
    from modules.ai import proposal_narrative as pn

    ctx = pn.ProposalContext(
        customer_name="T", site_address="A", system_size_kw=100.0,
        battery_capacity_kwh=0.0, nem_regime="NEM-2",
        year1_savings_usd=10_000.0, year1_bill_without_solar_usd=30_000.0,
        year1_bill_with_solar_usd=20_000.0, savings_pct=33.3,
        horizon_years=15, total_projected_savings_usd=200_000.0,
    )

    with patch.object(pn, "call_with_cache",
                      return_value=_text_message("- only one bullet")):
        bullets = pn.generate_executive_summary(ctx)

    assert len(bullets) >= 3  # fallback fires
    assert any("Year-1 savings" in b for b in bullets)


# ---------------------------------------------------------------------------
# bill_ingest
# ---------------------------------------------------------------------------
def test_bill_ingest_tool_schema_shape():
    from modules.ai.bill_ingest import TOOL_SCHEMA, TOOL_NAME

    assert TOOL_SCHEMA["name"] == TOOL_NAME
    props = TOOL_SCHEMA["input_schema"]["properties"]
    for key in ["utility", "rate_schedule", "total_kwh", "total_charges_usd"]:
        assert key in props
    assert set(TOOL_SCHEMA["input_schema"]["required"]) >= {
        "utility", "rate_schedule", "total_kwh", "total_charges_usd",
    }


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_bill_ingest_extracts_tool_input():
    from modules.ai import bill_ingest

    tool_input = {
        "utility": "PG&E", "rate_schedule": "AG-C",
        "billing_period_start": "2026-02-01",
        "billing_period_end": "2026-02-28",
        "total_kwh": 12_345.0, "peak_kwh": 2_000.0,
        "off_peak_kwh": 10_345.0, "max_demand_kw": 87.5,
        "total_charges_usd": 4_321.00, "nem_true_up": False,
    }

    with patch.object(bill_ingest, "_extract_pdf_text", return_value="raw pdf text"), \
         patch.object(bill_ingest, "call_with_cache",
                      return_value=_tool_message(bill_ingest.TOOL_NAME, tool_input)):
        out = bill_ingest.extract_bill(b"\x00fake pdf")

    assert out.utility == "PG&E"
    assert out.rate_schedule == "AG-C"
    assert out.total_kwh == pytest.approx(12_345.0)
    assert out.total_charges_usd == pytest.approx(4_321.00)
    assert out.nem_true_up is False
    assert out.raw_text == "raw pdf text"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_bill_ingest_handles_missing_tool_call():
    """Model refused to call the tool — extractor returns a BillExtraction
    with None fields rather than crashing."""
    from modules.ai import bill_ingest

    with patch.object(bill_ingest, "_extract_pdf_text", return_value="raw"), \
         patch.object(bill_ingest, "call_with_cache",
                      return_value=_text_message("I can't read this bill.")):
        out = bill_ingest.extract_bill(b"\x00")

    assert out.utility is None
    assert out.total_kwh is None
    assert out.nem_true_up is False


# ---------------------------------------------------------------------------
# tariff_qa
# ---------------------------------------------------------------------------
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_tariff_qa_sends_urdb_as_cached_context():
    from modules.ai import tariff_qa

    captured = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _text_message("  Peak hours are 16-21.  ")

    with patch.object(tariff_qa, "call_with_cache", side_effect=_fake_call):
        answer = tariff_qa.ask(
            "What are the peak hours?",
            tariff_label="PG&E E-19",
            urdb_json={"label": "e19", "energyweekdayschedule": [[0] * 24] * 12},
        )

    assert answer == "Peak hours are 16-21."
    blocks = captured["cached_context"]
    assert len(blocks) == 1
    assert blocks[0].kind == "text"
    assert "energyweekdayschedule" in blocks[0].content
    assert "PG&E E-19" in captured["user"]
