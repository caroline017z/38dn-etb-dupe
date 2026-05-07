"""Tests for modules.proposals — frozen dataclasses, round-trip serialisation,
migration, and isolation from the source ``saved_ppa_scenarios`` after snapshot.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.proposals import (
    MAX_COMPARISON_PPAS,
    PPASnapshot,
    Proposal,
    create_proposal,
    from_dict,
    list_proposals_in_session,
    migrate,
    save_proposal_to_session,
    snapshot_from_saved,
    to_dict,
    update_proposal,
)


def _sample_snapshot(name: str = "Base", term: int = 25) -> PPASnapshot:
    return PPASnapshot(
        name=name,
        rate_per_year=tuple(0.10 + 0.003 * i for i in range(term)),
        year1_rate_r1=0.10,
        year1_rate_r2=0.08,
        escalator_r1_pct=2.9,
        escalator_r2_pct=1.5,
        savings_pct=10.0,
        lifetime_savings_usd=1_234_567.0,
        nem_regime_1="NEM-2",
        nem_regime_2="NEM-3",
        num_years_1=5,
        term_years=term,
        calendar_years=tuple(2026 + i for i in range(term)),
        solar_kwh_per_year=tuple(450_000.0 * (0.995 ** i) for i in range(term)),
        utility_only_bill_k_per_year=tuple(80.0 + i for i in range(term)),
        total_ppa_bill_k_per_year=tuple(60.0 + i for i in range(term)),
    )


def _sample_proposal(comps: int = 1) -> Proposal:
    return create_proposal(
        name="WIC — Standard Offer",
        simulation_name="west-island-cotton-q1",
        customer_name="West Island Cotton",
        site_address="123 Gin Rd, Fresno CA",
        utility_account="PGE-12345",
        term_years=25,
        primary_ppa=_sample_snapshot("Primary"),
        comparison_ppas=tuple(_sample_snapshot(f"Alt-{i}") for i in range(comps)),
        narrative_bullets=("Year-1 savings ~$45k", "20-yr NPV ~$1.2MM"),
        notes="pref-rate on the table until Feb 28",
    )


# ---------------------------------------------------------------------------
# 1. Roundtrip: every field preserved through to_dict -> from_dict
# ---------------------------------------------------------------------------
def test_proposal_roundtrip_preserves_every_field():
    original = _sample_proposal(comps=3)
    payload = to_dict(original)
    import json
    # JSON must be a true round-trip, not just Python equality on dicts.
    rehydrated = from_dict(json.loads(json.dumps(payload)))

    # Scalar/string fields.
    for field_ in [
        "id", "schema_version", "name", "simulation_name", "customer_name",
        "site_address", "utility_account", "term_years", "created_at",
        "updated_at", "notes",
    ]:
        assert getattr(original, field_) == getattr(rehydrated, field_), field_

    assert rehydrated.narrative_bullets == original.narrative_bullets
    assert rehydrated.primary_ppa == original.primary_ppa
    assert rehydrated.comparison_ppas == original.comparison_ppas


# ---------------------------------------------------------------------------
# 2. Migration from v0 defaults the schema version
# ---------------------------------------------------------------------------
def test_migrate_from_v0_adds_schema_version():
    payload = to_dict(_sample_proposal(comps=0))
    del payload["schema_version"]
    payload.pop("comparison_ppas", None)
    payload.pop("narrative_bullets", None)
    payload.pop("notes", None)

    migrated = migrate(payload)
    assert migrated["schema_version"] == 1
    assert migrated["comparison_ppas"] == []
    assert migrated["narrative_bullets"] == []
    assert migrated["notes"] == ""

    # And from_dict still works through it end-to-end.
    parsed = from_dict(migrated)
    assert parsed.schema_version == 1


# ---------------------------------------------------------------------------
# 3. PPASnapshot is frozen (cannot mutate after creation)
# ---------------------------------------------------------------------------
def test_ppa_snapshot_is_frozen():
    snap = _sample_snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.name = "mutated"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.year1_rate_r1 = 0.99  # type: ignore[misc]


def test_proposal_is_frozen():
    p = _sample_proposal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "nope"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Old simulation dicts without a "proposals" key are tolerated
# ---------------------------------------------------------------------------
def test_old_simulation_json_without_proposals_loads_clean():
    # Simulates a session_state produced by loading an old simulation JSON
    # that predates the Proposals feature. No "proposals" key, no
    # "active_proposal_id" key.
    session = {"load_8760": object(), "tariff": object()}  # stand-ins
    assert list_proposals_in_session(session) == []
    assert session.get("proposals") is None  # reader must be non-mutating


# ---------------------------------------------------------------------------
# 5. Max 3 comparison PPAs — both create and update paths enforced
# ---------------------------------------------------------------------------
def test_create_proposal_rejects_more_than_three_comparisons():
    snaps = tuple(_sample_snapshot(f"x{i}") for i in range(MAX_COMPARISON_PPAS + 1))
    with pytest.raises(ValueError, match="at most"):
        create_proposal(
            name="too-many",
            simulation_name="s",
            customer_name="c",
            site_address="a",
            utility_account="u",
            term_years=25,
            primary_ppa=_sample_snapshot("primary"),
            comparison_ppas=snaps,
        )


def test_update_proposal_rejects_exceeding_comparison_cap():
    p = _sample_proposal(comps=0)
    snaps = tuple(_sample_snapshot(f"x{i}") for i in range(MAX_COMPARISON_PPAS + 1))
    with pytest.raises(ValueError):
        update_proposal(p, comparison_ppas=snaps)


# ---------------------------------------------------------------------------
# 6. Snapshot independence: mutating the source saved_ppa_scenarios dict
#    does not leak into a Proposal that already contains a snapshot.
# ---------------------------------------------------------------------------
def test_snapshot_is_independent_of_source_dict():
    saved = {
        "Base": {
            "ppa_rate_per_year": [0.10, 0.11, 0.12],
            "year1_rate_r1": 0.10,
            "year1_rate_r2": None,
            "ppa_escalator_r1": 2.9,
            "ppa_escalator_r2": None,
            "savings_pct": 10.0,
            "lifetime_savings": 100_000.0,
            "nem_regime_1": "NEM-3",
            "nem_regime_2": None,
            "num_years_1": None,
            "term_years": 3,
            "calendar_year": [2026, 2027, 2028],
        },
    }
    snap = snapshot_from_saved("Base", saved["Base"], term_years=3)

    # Mutate the source dict heavily — snapshot must remain untouched.
    saved["Base"]["year1_rate_r1"] = 999.0
    saved["Base"]["ppa_rate_per_year"] = [42.0, 42.0, 42.0]
    saved["Base"]["lifetime_savings"] = -1.0
    del saved["Base"]["calendar_year"]

    assert snap.year1_rate_r1 == 0.10
    assert snap.rate_per_year == (0.10, 0.11, 0.12)
    assert snap.lifetime_savings_usd == 100_000.0
    assert snap.calendar_years == (2026, 2027, 2028)


# ---------------------------------------------------------------------------
# Extra: session helpers round-trip
# ---------------------------------------------------------------------------
def test_save_and_list_in_session():
    session: dict = {}
    p = _sample_proposal()
    save_proposal_to_session(session, p)
    assert session["active_proposal_id"] == p.id
    found = list_proposals_in_session(session, simulation_name=p.simulation_name)
    assert [x.id for x in found] == [p.id]
    # Scoping filter excludes mismatched simulations.
    assert list_proposals_in_session(session, simulation_name="other-sim") == []
