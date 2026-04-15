"""Shared pytest configuration.

``PV_SIM_REGEN_GOLDEN=1`` (env var) or ``--regen-golden`` (CLI flag) causes
golden regression tests to rewrite their expected-output JSON files instead
of asserting. Use when legitimate billing-engine changes shift the numbers;
review the diff, commit the new expected files, and run again without the
flag to confirm stability.
"""

from __future__ import annotations

import os


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Regenerate golden expected-output files instead of asserting",
    )


def pytest_configure(config):
    if config.getoption("--regen-golden"):
        os.environ["PV_SIM_REGEN_GOLDEN"] = "1"
