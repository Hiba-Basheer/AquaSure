import pandas as pd
from src.preprocessing import preprocess


def test_preprocessing_outputs():
    dummy_data = pd.DataFrame({
        "ph": [7.0, 8.0] * 5,
        "Hardness": [200, 180] * 5,
        "Solids": [10000, 12000] * 5,
        "Chloramines": [7.0, 6.5] * 5,
        "Sulfate": [300, 280] * 5,
        "Conductivity": [400, 420] * 5,
        "Organic_carbon": [10, 11] * 5,
        "Trihalomethanes": [80, 75] * 5,
        "Turbidity": [3.0, 3.5] * 5,
        "Potability": [1, 0] * 5,
    })

    X_train, X_test, y_train, y_test = preprocess(dummy_data)

    # Check shapes of the split data
    # With 10 samples, 20% test size results in 2 test, 8 train.
    # train_test_split usually tries to respect the ratio.
    # Let's just check that we got return values and they aren't empty.
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    # Since we have 10 samples and test_size=0.2, usually 2 in test, 8 in train.
    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10