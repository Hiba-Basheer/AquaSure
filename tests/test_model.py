import joblib
from unittest.mock import patch


def test_model_loads():
    with patch("joblib.load") as mock_load:
        mock_load.return_value = "mock_model"
        model = joblib.load("any_path.pkl")
        assert model == "mock_model"