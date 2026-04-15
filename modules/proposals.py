"""Customer Proposal objects — named containers that group a simulation,
a primary PPA, and up to 3 comparison PPAs.

Design invariants (locked down by the Phase 4 architecture review):
- **Snapshot, don't link**: when a PPA is added to a Proposal, its rate
  schedule and metadata are *copied* into a :class:`PPASnapshot`. Later
  edits or deletions of the source PPA in ``saved_ppa_scenarios`` do not
  mutate the Proposal.
- **One Proposal is scoped to one simulation** (``simulation_name``) but
  may hold multiple PPA structure options for side-by-side comparison.
- **Frozen dataclasses everywhere** so Monte Carlo / export code can
  share instances without fear of mutation.
- **Pure vs I/O split**: :func:`create_proposal` and friends return new
  dataclass instances. Session-state + GCS I/O live in
  :func:`save_proposal_to_session` and :func:`persist_proposal_to_gcs`.
- **Explicit schema_version on every serialized payload** so we can add
  fields without breaking loads of older Proposals.

Naming disambiguation: the word "Proposal" here refers to the *container*
(this module). The PPTX that gets sent to a customer is the **Customer
Proposal Deck** — that's what ``modules/proposal.py::generate_proposal_pptx``
produces.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
MAX_COMPARISON_PPAS = 3

# Local + GCS persistence directory naming.
LOCAL_PROPOSALS_DIR = "data/proposals"
GCS_PROPOSALS_PREFIX = "proposals/"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PPASnapshot:
    """A point-in-time copy of a saved PPA scenario, inlined into a Proposal.

    Lists are serialised as ``tuple`` for frozen-hashability. The mirror
    structure in ``st.session_state["saved_ppa_scenarios"]`` uses ``list``
    — see :func:`snapshot_from_saved` for the bridge.
    """

    name: str
    # PPA rate per year for the whole term (len = term_years).
    rate_per_year: tuple[float, ...]
    # Per-regime year-1 rates — regime_2 is None when no NEM switch.
    year1_rate_r1: float
    year1_rate_r2: float | None
    escalator_r1_pct: float
    escalator_r2_pct: float | None
    savings_pct: float
    lifetime_savings_usd: float
    nem_regime_1: str
    nem_regime_2: str | None
    num_years_1: int | None
    term_years: int
    # Auxiliary arrays used by the comparison chart — may be None if the
    # source scenario predates the field.
    calendar_years: tuple[int, ...] | None = None
    solar_kwh_per_year: tuple[float, ...] | None = None
    utility_only_bill_k_per_year: tuple[float, ...] | None = None
    total_ppa_bill_k_per_year: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Proposal:
    """A named customer-facing bundle.

    One Proposal ↔ one simulation, many PPA structure options (primary +
    up to 3 comparisons).
    """

    id: str                                   # uuid4 hex
    schema_version: int                       # migration key
    name: str                                 # user-supplied
    simulation_name: str | None               # scopes the Proposal
    customer_name: str
    site_address: str
    utility_account: str
    term_years: int
    primary_ppa: PPASnapshot
    comparison_ppas: tuple[PPASnapshot, ...]  # len 0..MAX_COMPARISON_PPAS
    narrative_bullets: tuple[str, ...]
    created_at: str                           # ISO8601 (JSON-safe)
    updated_at: str
    # Free-form analyst notes; empty string when absent.
    notes: str = ""


# ---------------------------------------------------------------------------
# Snapshot bridge (saved_ppa_scenarios dict → PPASnapshot)
# ---------------------------------------------------------------------------
def snapshot_from_saved(name: str, saved_dict: dict, *, term_years: int) -> PPASnapshot:
    """Extract a :class:`PPASnapshot` from the ``saved_ppa_scenarios[name]``
    structure already persisted by the PPA Rate tab.

    The PPA-tab save format has evolved across phases; this bridge tolerates
    missing fields and fills with safe defaults.
    """
    def _tuple(v):
        return tuple(v) if v is not None else None

    rate_per_year = _tuple(saved_dict.get("ppa_rate_per_year")) or ()
    # Older saved scenarios used "year1_rate" for single-regime.
    y1_r1 = saved_dict.get("year1_rate_r1", saved_dict.get("year1_rate"))
    y1_r2 = saved_dict.get("year1_rate_r2")
    return PPASnapshot(
        name=name,
        rate_per_year=rate_per_year,
        year1_rate_r1=float(y1_r1 or 0.0),
        year1_rate_r2=(float(y1_r2) if y1_r2 is not None else None),
        escalator_r1_pct=float(saved_dict.get("ppa_escalator_r1") or 0.0),
        escalator_r2_pct=(
            float(saved_dict["ppa_escalator_r2"])
            if saved_dict.get("ppa_escalator_r2") is not None else None
        ),
        savings_pct=float(saved_dict.get("savings_pct") or 0.0),
        lifetime_savings_usd=float(saved_dict.get("lifetime_savings") or 0.0),
        nem_regime_1=str(saved_dict.get("nem_regime_1") or ""),
        nem_regime_2=(
            str(saved_dict["nem_regime_2"])
            if saved_dict.get("nem_regime_2") else None
        ),
        num_years_1=(
            int(saved_dict["num_years_1"])
            if saved_dict.get("num_years_1") is not None else None
        ),
        term_years=int(saved_dict.get("term_years") or term_years or len(rate_per_year) or 25),
        calendar_years=_tuple(saved_dict.get("calendar_year")),
        solar_kwh_per_year=_tuple(saved_dict.get("solar_kwh_per_year")),
        utility_only_bill_k_per_year=_tuple(saved_dict.get("utility_only_bill_k")),
        total_ppa_bill_k_per_year=_tuple(saved_dict.get("total_ppa_bill_k")),
    )


# ---------------------------------------------------------------------------
# Pure CRUD — no session_state, no I/O
# ---------------------------------------------------------------------------
def create_proposal(
    *,
    name: str,
    simulation_name: str | None,
    customer_name: str,
    site_address: str,
    utility_account: str,
    term_years: int,
    primary_ppa: PPASnapshot,
    comparison_ppas: tuple[PPASnapshot, ...] = (),
    narrative_bullets: tuple[str, ...] = (),
    notes: str = "",
    now: datetime | None = None,
) -> Proposal:
    """Return a new Proposal. Raises ``ValueError`` on >3 comparisons."""
    if len(comparison_ppas) > MAX_COMPARISON_PPAS:
        raise ValueError(
            f"A Proposal may hold at most {MAX_COMPARISON_PPAS} comparison PPAs; "
            f"got {len(comparison_ppas)}."
        )
    now = now or datetime.now()
    ts = now.isoformat(timespec="seconds")
    return Proposal(
        id=uuid.uuid4().hex,
        schema_version=SCHEMA_VERSION,
        name=name,
        simulation_name=simulation_name,
        customer_name=customer_name,
        site_address=site_address,
        utility_account=utility_account,
        term_years=int(term_years),
        primary_ppa=primary_ppa,
        comparison_ppas=tuple(comparison_ppas),
        narrative_bullets=tuple(narrative_bullets),
        created_at=ts,
        updated_at=ts,
        notes=notes,
    )


def update_proposal(proposal: Proposal, **fields) -> Proposal:
    """Return a copy with the requested fields patched; refuses fields that
    aren't on :class:`Proposal`. Touches ``updated_at``."""
    allowed = set(Proposal.__dataclass_fields__) - {"id", "created_at", "schema_version"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown or immutable Proposal fields: {sorted(unknown)}")
    if "comparison_ppas" in fields and len(fields["comparison_ppas"]) > MAX_COMPARISON_PPAS:
        raise ValueError(f"At most {MAX_COMPARISON_PPAS} comparison PPAs are allowed")
    return replace(
        proposal,
        **fields,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Serialisation (JSON-safe dicts; migrations live here)
# ---------------------------------------------------------------------------
def to_dict(proposal: Proposal) -> dict:
    """JSON-safe dict. Tuples become lists; no datetime objects."""
    return _coerce(asdict(proposal))


def _coerce(obj):
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    return obj


def from_dict(data: dict) -> Proposal:
    """Parse a dict into a :class:`Proposal`, running migrations first.

    ``data`` is never mutated — we always work on a copy.
    """
    migrated = migrate(dict(data))
    primary = _snapshot_from_dict(migrated["primary_ppa"])
    comps = tuple(_snapshot_from_dict(c) for c in (migrated.get("comparison_ppas") or ()))
    return Proposal(
        id=str(migrated["id"]),
        schema_version=int(migrated["schema_version"]),
        name=str(migrated.get("name", "")),
        simulation_name=migrated.get("simulation_name"),
        customer_name=str(migrated.get("customer_name", "")),
        site_address=str(migrated.get("site_address", "")),
        utility_account=str(migrated.get("utility_account", "")),
        term_years=int(migrated.get("term_years", 25)),
        primary_ppa=primary,
        comparison_ppas=comps,
        narrative_bullets=tuple(migrated.get("narrative_bullets") or ()),
        created_at=str(migrated.get("created_at", "")),
        updated_at=str(migrated.get("updated_at", migrated.get("created_at", ""))),
        notes=str(migrated.get("notes", "")),
    )


def _snapshot_from_dict(data: dict) -> PPASnapshot:
    def _tuple(v):
        return tuple(v) if v else None

    return PPASnapshot(
        name=str(data.get("name", "")),
        rate_per_year=tuple(data.get("rate_per_year") or ()),
        year1_rate_r1=float(data.get("year1_rate_r1") or 0.0),
        year1_rate_r2=(float(data["year1_rate_r2"])
                       if data.get("year1_rate_r2") is not None else None),
        escalator_r1_pct=float(data.get("escalator_r1_pct") or 0.0),
        escalator_r2_pct=(float(data["escalator_r2_pct"])
                          if data.get("escalator_r2_pct") is not None else None),
        savings_pct=float(data.get("savings_pct") or 0.0),
        lifetime_savings_usd=float(data.get("lifetime_savings_usd") or 0.0),
        nem_regime_1=str(data.get("nem_regime_1", "")),
        nem_regime_2=(str(data["nem_regime_2"]) if data.get("nem_regime_2") else None),
        num_years_1=(int(data["num_years_1"])
                     if data.get("num_years_1") is not None else None),
        term_years=int(data.get("term_years") or 25),
        calendar_years=_tuple(data.get("calendar_years")),
        solar_kwh_per_year=_tuple(data.get("solar_kwh_per_year")),
        utility_only_bill_k_per_year=_tuple(data.get("utility_only_bill_k_per_year")),
        total_ppa_bill_k_per_year=_tuple(data.get("total_ppa_bill_k_per_year")),
    )


def migrate(data: dict) -> dict:
    """Bring an older Proposal dict up to the current schema.

    Missing ``schema_version`` means the payload was written by a pre-v1
    codepath (none exist today, but the hook is here so future migrations
    stay additive).
    """
    v = int(data.get("schema_version") or 0)
    if v < 1:
        # v0 -> v1: ensure tuple fields exist.
        data.setdefault("comparison_ppas", [])
        data.setdefault("narrative_bullets", [])
        data.setdefault("notes", "")
        data["schema_version"] = 1
    # Future: if v < 2: ...
    return data


# ---------------------------------------------------------------------------
# Session-state I/O (thin — pure logic above does the real work)
# ---------------------------------------------------------------------------
SESSION_KEY_PROPOSALS = "proposals"
SESSION_KEY_ACTIVE = "active_proposal_id"


def save_proposal_to_session(session_state, proposal: Proposal) -> None:
    store = session_state.setdefault(SESSION_KEY_PROPOSALS, {})
    store[proposal.id] = to_dict(proposal)
    session_state[SESSION_KEY_ACTIVE] = proposal.id


def delete_proposal_from_session(session_state, proposal_id: str) -> None:
    store = session_state.get(SESSION_KEY_PROPOSALS) or {}
    store.pop(proposal_id, None)
    if session_state.get(SESSION_KEY_ACTIVE) == proposal_id:
        remaining = list(store.keys())
        session_state[SESSION_KEY_ACTIVE] = remaining[0] if remaining else None


def list_proposals_in_session(
    session_state,
    simulation_name: str | None = None,
) -> list[Proposal]:
    """Return Proposals from ``session_state["proposals"]`` optionally
    filtered by simulation name. Skips any that fail to parse (logged).
    """
    store = session_state.get(SESSION_KEY_PROPOSALS) or {}
    out: list[Proposal] = []
    for pid, raw in store.items():
        try:
            p = from_dict(raw)
        except Exception as exc:  # noqa: BLE001 — resilience over strictness
            logger.warning("failed to parse proposal %s: %s", pid, exc)
            continue
        if simulation_name is None or p.simulation_name == simulation_name:
            out.append(p)
    out.sort(key=lambda p: p.updated_at, reverse=True)
    return out


def get_active_proposal(session_state) -> Proposal | None:
    pid = session_state.get(SESSION_KEY_ACTIVE)
    if not pid:
        return None
    store = session_state.get(SESSION_KEY_PROPOSALS) or {}
    raw = store.get(pid)
    if not raw:
        return None
    try:
        return from_dict(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("active proposal %s failed to parse: %s", pid, exc)
        return None


# ---------------------------------------------------------------------------
# Persistence — local + GCS, using the sim_helpers primitives
# ---------------------------------------------------------------------------
def _local_proposal_dir(simulation_name: str | None) -> str:
    sub = simulation_name if simulation_name else "_unscoped"
    return os.path.join(LOCAL_PROPOSALS_DIR, sub)


def _gcs_prefix(simulation_name: str | None) -> str:
    sub = simulation_name if simulation_name else "_unscoped"
    return f"{GCS_PROPOSALS_PREFIX}{sub}/"


def persist_proposal(proposal: Proposal) -> None:
    """Write the Proposal to local disk + GCS.

    Late import of :mod:`sim_helpers` so this module stays importable in
    isolated tests without Streamlit or Google-cloud-storage.
    """
    from sim_helpers import save_profile_bytes, sanitize_filename

    payload = json.dumps(to_dict(proposal), indent=2).encode("utf-8")
    save_profile_bytes(
        local_dir=_local_proposal_dir(proposal.simulation_name),
        gcs_prefix=_gcs_prefix(proposal.simulation_name),
        name=sanitize_filename(proposal.id),
        data=payload,
        ext=".json",
    )


def load_proposals_for_simulation(simulation_name: str | None) -> list[Proposal]:
    """Load every Proposal stored under ``simulation_name`` (local + GCS)."""
    from sim_helpers import list_profile_files, load_profile_bytes

    local_dir = _local_proposal_dir(simulation_name)
    prefix = _gcs_prefix(simulation_name)
    names = list_profile_files(local_dir, prefix, ".json")
    out: list[Proposal] = []
    for n in names:
        raw = load_profile_bytes(local_dir, prefix, n, ".json")
        if not raw:
            continue
        try:
            out.append(from_dict(json.loads(raw.decode("utf-8"))))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping un-parseable proposal %s: %s", n, exc)
    out.sort(key=lambda p: p.updated_at, reverse=True)
    return out


def delete_persisted_proposal(proposal: Proposal) -> None:
    from sim_helpers import delete_profile_file, sanitize_filename

    delete_profile_file(
        local_dir=_local_proposal_dir(proposal.simulation_name),
        gcs_prefix=_gcs_prefix(proposal.simulation_name),
        name=sanitize_filename(proposal.id),
        ext=".json",
    )


__all__ = [
    "SCHEMA_VERSION",
    "MAX_COMPARISON_PPAS",
    "PPASnapshot",
    "Proposal",
    "SESSION_KEY_PROPOSALS",
    "SESSION_KEY_ACTIVE",
    "snapshot_from_saved",
    "create_proposal",
    "update_proposal",
    "to_dict",
    "from_dict",
    "migrate",
    "save_proposal_to_session",
    "delete_proposal_from_session",
    "list_proposals_in_session",
    "get_active_proposal",
    "persist_proposal",
    "load_proposals_for_simulation",
    "delete_persisted_proposal",
]
