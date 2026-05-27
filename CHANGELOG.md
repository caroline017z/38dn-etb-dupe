# Changelog

All notable changes to PV Solar Rate Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Default billing option is now **Monthly (MBO)** instead of Annual (ABO).

### Fixed
- **Credit offsets energy only.** Generation/export credits no longer reduce demand, fixed/customer, or non-bypassable (NBC) charges — only the volumetric energy charge — across all bill-assembly paths (NEM-1/2 MBO/ABO, NEM-3, ECC engine, NEM-A aggregation) and the battery size-selection objective. Per PG&E NEM2 SC 2.c/2.d and Schedule NBT SC 2.d.
- **NEM-2 NBC double-count.** The URDB retail rate already includes the non-bypassable components, so NEM-2 energy is now netted at the rate *excluding* `nbc_rate`, with NBC charged once via the explicit non-bypassable line. Exports are credited net of NBC (they do not earn the non-bypassable portion). Increases reported savings for NEM-2 deals with a positive NBC rate; the `tou_nem2` golden was regenerated accordingly.
- **NEM-A benefiting meters now pay NBC** on their own net consumption (was zeroed); NBCs are non-bypassable per metered account.
- **Projection tie-out under degradation.** The monthly and annual projection builders now share one degradation-volume model, so the views reconcile when `degradation_pct > 0`.
- **ECC engine** now applies the NEM-3/NBT year-end NSC clawback (previously always a no-op) and floors the monthly bill at the larger of the minimum and fixed charges.
- **Y>1 NBC** now escalates with import volume, matching demand/energy.

## [1.0.0] - 2026-02-10

### Added

#### Billing Engines
- Dual billing engines: custom TOU/demand/export engine and ECC (electricitycostcalculator) adapter
- Support for California IOUs (PG&E, SCE, SDG&E) via OpenEI URDB tariff data

#### NEM Regime Modeling
- NEM-1 support with time-of-use netting
- NEM-2 support with TOU netting plus non-bypassable charges (NBC)
- NEM-3 / Net Value Billing Tariff (NVBT) support with hourly settlement
- Mid-life NEM regime switching for multi-year projections
- Regime-aware energy cost calculation: TOU-netted for NEM-1/NEM-2, raw import for NEM-3

#### Solar Production
- PVWatts v8 API integration with address geocoding for 8760 solar production profiles
- Advanced PV options: module type (Standard, Premium, Thin Film), system losses, annual degradation

#### Battery Storage
- Battery storage dispatch optimization via CVXPY linear programming (annual and monthly modes)
- Automatic battery capacity sizing sweep

#### Demand and Export
- Demand charge calculation with flat and TOU-period monthly demand charges
- Export compensation via ACC 8760 hourly profiles, flat rates, or user-uploaded CSVs
- ACC rate year indexing aligned to calendar year with escalation beyond CSV range

#### Financial Projections
- Multi-year financial projections with rate escalators, load escalators, and solar degradation

#### Outputs and Reports
- Interactive Plotly charts and styled summary tables
- Downloadable Excel and CSV reports (hourly, monthly, annual)
- Reorganized result tabs: Monthly Bills, Annual Projection, PPA Rate, Downloads

#### Simulation Management
- Simulation save, load, compare, and delete management with full input/output persistence
- Load profile and export profile upload and persistence

#### User Experience
- Getting-started guidance and simulation checklist for new users
- Sidebar tooltips and help text on all major inputs
- Improved error messages with user-friendly troubleshooting and collapsible tracebacks
