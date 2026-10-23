"""API contracts with a synthetic predictor; no saved models or datasets read."""

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("prediction_api", Path(__file__).parents[1] / "model/predict.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class Predictor:
    def __init__(self, values=None):
        self.values = values
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return pd.Series(self.values if self.values is not None else [12.5] * len(frame))


@pytest.fixture
def client():
    predictor = Predictor()
    with TestClient(api.create_app(predictor_loader=lambda path: predictor)) as client:
        yield client, predictor


def test_default_model_path_matches_training_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert api.resolve_model_path("models/model") == api.PROJECT_ROOT / "models/model"
    assert api.resolve_model_path(tmp_path / "model") == tmp_path / "model"
    assert api.MODEL_PATH == "models/model"


def test_unavailable_model_keeps_api_discoverable():
    def missing(path):
        raise FileNotFoundError("private/model/path")
    with TestClient(api.create_app(predictor_loader=missing)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/health").status_code == 503
        response = client.post("/predict", json={"area": 12})
        assert response.status_code == 503
        assert "private/model/path" not in response.text


def test_single_and_both_batch_endpoints_have_stable_shapes(client):
    client, predictor = client
    assert client.get("/health").json()["model_loaded"] is True
    assert client.post("/predict", json={"area": 10}).json() == {"prediction": 12.5}
    for url, body in [("/predict", {"records": [{"area": 1}]}),
                      ("/predict/batch", [{"area": 1}])]:
        assert client.post(url, json=body).json() == {"predictions": [12.5], "count": 1}
    assert client.post("/predict", json={"records": [{"area": 1}, {"area": 2}]}).json() == {
        "predictions": [12.5, 12.5], "count": 2}
    assert predictor.frames[-1]["area"].tolist() == [1, 2]


@pytest.mark.parametrize("records", [[], {}, [None], [{}], [1], [{"area": 1}] * 1001])
def test_bad_batches_rejected_before_inference(client, records):
    client, predictor = client
    assert client.post("/predict", json={"records": records}).status_code == 400
    assert not predictor.frames


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [], [1, 2], [True], ["bad"]])
def test_invalid_model_output_is_an_explicit_failure(values):
    with TestClient(api.create_app(predictor_loader=lambda path: Predictor(values))) as client:
        response = client.post("/predict", json={"area": 10})
        assert response.status_code == 500


def test_feature_failure_is_sanitized():
    class Missing:
        def predict(self, frame):
            raise KeyError("private internal feature/path")
    with TestClient(api.create_app(predictor_loader=lambda path: Missing())) as client:
        response = client.post("/predict", json={"other": 1})
        assert response.status_code == 400
        assert "private internal" not in response.text


def test_cli_csv_is_atomic_and_preserves_source(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    source.write_text("area\n1\n2\n")
    api.make_predictions(source, output, predictor_loader=lambda path: Predictor())
    assert pd.read_csv(output)["prediction"].tolist() == [12.5, 12.5]
    assert source.read_text() == "area\n1\n2\n"
    with pytest.raises(RuntimeError):
        api.make_predictions(source, output, predictor_loader=lambda path: Predictor([float("nan"), 3]))
    assert pd.read_csv(output)["prediction"].tolist() == [12.5, 12.5]
    assert not list(tmp_path.glob(".prediction-*"))


def test_help_does_not_require_autogluon():
    with pytest.raises(SystemExit) as error:
        api.main(["--help"])
    assert error.value.code == 0


def test_cli_refuses_source_alias_before_loading_model(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("area\n1\n")
    alias = tmp_path / "alias.csv"
    alias.hardlink_to(source)
    for output in [source, alias]:
        with pytest.raises(ValueError, match="different"):
            api.make_predictions(source, output, predictor_loader=lambda path: pytest.fail("must not load"))
    assert source.read_text() == "area\n1\n"
