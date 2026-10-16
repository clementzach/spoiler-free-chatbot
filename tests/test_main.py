import pytest

from app.main import app, check_config
from fastapi.testclient import TestClient


def test_check_config_requires_keys_for_selected_judge():
    check_config({"OPENAI_API_KEY": "x"})
    check_config({"OPENAI_API_KEY": "x", "JUDGE": "decisions"})
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_config({})
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        check_config({"OPENAI_API_KEY": "x", "JUDGE": "jev"})
    with pytest.raises(RuntimeError, match="JUDGE"):
        check_config({"OPENAI_API_KEY": "x", "JUDGE": "bogus"})


def test_healthz_and_index(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    with TestClient(app) as c:  # runs lifespan, i.e. the config check
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/").status_code == 200
