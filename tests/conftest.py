"""Shared fixtures and async test helper."""
from __future__ import annotations

import asyncio

import pytest

from mdbt.spec import MachineSpec, Transition, load_spec_file


SPEC_PATH = "fixtures/order_machine.json"


@pytest.fixture(scope="session")
def spec() -> MachineSpec:
    return load_spec_file(SPEC_PATH)


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "evidence.db")


@pytest.fixture
def replay_dir(tmp_path):
    p = tmp_path / "replays"
    p.mkdir()
    return str(p)
