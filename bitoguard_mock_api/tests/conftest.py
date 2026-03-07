from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    data_dir = Path(__file__).resolve().parents[2] / "bitoguard_sim_output"
    return TestClient(create_app(data_dir=data_dir))
