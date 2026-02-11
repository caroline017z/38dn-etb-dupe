# PV Solar Rate Simulator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

A Streamlit web application for simulating annual electricity bills for California agricultural and commercial customers with solar PV systems. Compare billing outcomes across NEM regimes, optimize battery storage dispatch, and project multi-year financial savings.

---

## Features

### Billing Engines
- **Dual billing engines** -- choose between a built-in custom TOU/demand/export engine or the [electricitycostcalculator](https://github.com/Breakthrough-Energy/electricitycostcalculator) (ECC) adapter backed by OpenEI tariff data
- **California IOU support** -- PG&E, SCE, and SDG&E rate schedules pulled live from the OpenEI Utility Rate Database (URDB)

### NEM Regime Modeling
- **NEM-1** -- time-of-use netting of solar generation against site load
- **NEM-2** -- TOU netting plus non-bypassable charges (NBC)
- **NEM-3 / Net Value Billing Tariff (NVBT)** -- hourly settlement with ACC-based export rates
- **Mid-life NEM regime switching** -- model a transition between NEM regimes partway through a system's financial life for multi-year projections
- **Regime-aware energy cost** -- TOU-netted costs for NEM-1/NEM-2, raw import costs for NEM-3

### Solar Production
- **PVWatts v8 API integration** -- generate 8760 hourly solar production profiles via the NREL PVWatts API with address geocoding
- **Advanced PV options** -- module type selection (Standard, Premium, Thin Film), configurable system losses, and annual degradation modeling

### Battery Storage
- **Dispatch optimization** -- co-located battery energy storage system (BESS) dispatch via CVXPY linear programming, with annual and monthly solve modes
- **Automatic capacity sizing sweep** -- evaluate a range of battery capacities to identify the optimal system size

### Demand and Export
- **Demand charge calculation** -- flat and TOU-period monthly demand charges computed from net import peaks
- **Export compensation** -- ACC 8760 hourly profiles, flat rates, or user-uploaded CSVs for export credit valuation
- **ACC rate year indexing** -- aligned to calendar year with automatic escalation beyond the CSV data range

### Financial Projections
- **Multi-year projections** -- rate escalators, load escalators, and solar degradation applied year over year
- **Annual and monthly breakdowns** -- detailed financial results at both time scales

### Outputs and Reports
- **Interactive Plotly charts** and styled summary tables across Monthly Bills, Annual Projection, PPA Rate, and Downloads tabs
- **Downloadable Excel and CSV reports** -- hourly, monthly, and annual data exports

### Simulation Management
- **Save, load, compare, and delete** simulation scenarios with full input/output persistence
- **Load profile and export profile upload** with session persistence

### User Experience
- **Getting-started guidance** and simulation checklist that highlights incomplete steps for new users
- **Sidebar tooltips and help text** on all major inputs

## Prerequisites

- **Python 3.10** or newer
- **NREL API key** -- free from <https://developer.nrel.gov/signup/>

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/pv-rate-sim.git
   cd pv-rate-sim
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example file and add your API keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your keys:

   ```
   NREL_API_KEY=your_nrel_api_key_here
   OPENEI_API_KEY=your_openei_api_key_here
   ```

## Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Basic Workflow

1. **Select a utility** (PG&E, SCE, or SDG&E) and fetch available rate schedules
2. **Enter site details** -- address for geocoding, annual load (kWh), and PV system parameters
3. **Upload or generate profiles** -- 8760 load profile (CSV) and solar production via PVWatts
4. **Choose a billing engine** -- Custom (built-in) or ECC (OpenEI-backed)
5. **Configure NEM regime** -- NEM-1, NEM-2, or NEM-3/NVBT with export rate source
6. **(Optional) Add battery storage** -- set capacity, efficiency, and charge/discharge windows, or let the optimizer size the system
7. **Run the simulation** and review results across four tabs: Monthly Bills, Annual Projection, PPA Rate, and Downloads
8. **Download results** as Excel or CSV, or save the simulation for later comparison

A getting-started guide and simulation checklist appear automatically to walk you through any steps that still need input.

## Project Structure

```
pv-rate-sim/
├── app.py                     # Main Streamlit application
├── sim_helpers.py             # Simulation save/load/delete utilities
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
│
├── modules/
│   ├── tariff.py              # OpenEI URDB tariff fetching and parsing
│   ├── pvwatts.py             # NREL PVWatts API integration
│   ├── billing.py             # Custom billing engine (TOU/demand/export)
│   ├── billing_ecc.py         # ECC library billing adapter
│   ├── demand.py              # Monthly demand charge calculations
│   ├── export_value.py        # ACC / flat / uploaded export rate handling
│   ├── outputs.py             # Charts, tables, CSV/Excel generation
│   └── battery/
│       ├── config.py          # BatteryConfig dataclass
│       ├── dispatch.py        # CVXPY LP dispatch optimizer
│       └── sizing.py          # Capacity sizing sweep
│
├── assets/                    # Logo images
├── data/                      # Runtime data (gitignored)
│   ├── simulations/           # Saved simulation JSON files
│   ├── load_profiles/         # Uploaded load profile CSVs
│   ├── export_profiles/       # Uploaded export rate CSVs
│   ├── ecc_tariffs/           # Cached ECC tariff JSON files
│   └── acc_export_rates/      # ACC 8760 export rate CSVs
│
└── tests/
    └── test_battery_dispatch.py  # Battery dispatch unit tests
```

## Environment Variables

| Variable | Source | Description |
|---|---|---|
| `NREL_API_KEY` | `.env` | Required. Your NREL Developer API key for PVWatts solar production queries. |
| `OPENEI_API_KEY` | `.env` | Optional. OpenEI API key for URDB tariff lookups (falls back to anonymous access). |

The `data/` directory is created automatically at runtime and stores user-generated profiles and simulations. It is excluded from version control.

## Running Tests

```bash
pytest tests/
```

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

Please ensure all existing tests pass before submitting.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [NREL PVWatts](https://pvwatts.nrel.gov/) for solar production modeling
- [OpenEI URDB](https://openei.org/wiki/Utility_Rate_Database) for utility tariff data
- [California Avoided Cost Calculator (ACC)](https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/energy-efficiency/avoided-cost-calculator) for hourly export rate data
- [electricitycostcalculator](https://github.com/Breakthrough-Energy/electricitycostcalculator) for the ECC billing engine
- [CVXPY](https://www.cvxpy.org/) for convex optimization
- [Streamlit](https://streamlit.io/) for the web application framework
