"""Shared fixtures. The container (model + cache + graph + case replay) is built once per test session."""
import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def container():
    from backend.app.deps import build_container, set_container
    c = build_container(db_path=":memory:")
    set_container(c)
    return c


@pytest.fixture(scope="session")
def client(container):
    from backend.app.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def history(container):
    return container.scoring.history


def raw_txn(row: dict, **over) -> dict:
    """Cache row -> JSON body for the API's Transaction model."""
    keys = ["user_id", "type", "amount", "recipient_id", "agent_id", "device_id", "location", "balance_before",
            "merchant_category", "otp_requests_10m", "otp_failures_10m", "otp_device_mismatch", "concurrent_sessions",
            "sim_swap_recent"]
    out = {k: (None if isinstance(row[k], float) and row[k] != row[k] else row[k]) for k in keys}
    out["ts"] = str(row["ts"]).replace(" ", "T")
    out.update(over)
    return out
