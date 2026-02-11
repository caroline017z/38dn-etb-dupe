# Changelog

All notable changes to PV Solar Rate Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
