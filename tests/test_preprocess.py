import pytest
from lib_ml_remla import split_data, preprocess_data
import numpy as np

# Sample data for testing
@pytest.fixture
def sample_data():
    train = ["0\tThe quick brown fox", "1\tjumps over the lazy dog"]
    test = ["0\tLorem ipsum dolor sit", "1\tamet, consectetur adipiscing elit"]
    val = ["0\tSed do eiusmod tempor", "1\tincididunt ut labore et dolore magna aliqua"]
    return train, test, val

def test_split_data(sample_data):
    train, test, val = sample_data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(train, test, val)

    assert X_train == ["The quick brown fox", "jumps over the lazy dog"]
    assert y_train == ["0", "1"]
    assert X_test == ["Lorem ipsum dolor sit", "amet, consectetur adipiscing elit"]
    assert y_test == ["0", "1"]
    assert X_val == ["Sed do eiusmod tempor", "incididunt ut labore et dolore magna aliqua"]
    assert y_val == ["0", "1"]

def test_preprocess_data(sample_data):
    train, test, val = sample_data
    _, y_train, _, y_val, _, y_test = split_data(train, test, val)
    X_train, y_train, X_val, y_val, X_test, y_test, char_index, tokenizer, encoder = preprocess_data(
        ["The quick brown fox", "jumps over the lazy dog"],
        y_train,
        ["Sed do eiusmod tempor", "incididunt ut labore et dolore magna aliqua"],
        y_val,
        ["Lorem ipsum dolor sit", "amet, consectetur adipiscing elit"],
        y_test
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert len(char_index) > 0  # Check if tokenizer created a character index
    assert X_train.shape == (2, 200)  # Assuming sequence_length is 200
    assert X_val.shape == (2, 200)
    assert X_test.shape == (2, 200)
    assert y_train.shape == (2,)
    assert y_val.shape == (2,)
    assert y_test.shape == (2,)
