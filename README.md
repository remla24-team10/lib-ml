<!-- # package_template_tester -->

<!-- <div align="center"> -->

<!-- [![Build status](https://github.com/test/package_template_tester/workflows/build/badge.svg?branch=master&event=push)](https://github.com/test/package_template_tester/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/package_template_tester.svg)](https://pypi.org/project/package_template_tester/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/test/package_template_tester/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/test/package_template_tester/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/test/package_template_tester/releases)
[![License](https://img.shields.io/github/license/test/package_template_tester)](https://github.com/test/package_template_tester/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg) -->


<!-- </div>

## Installation
> Python 3.11 is needed for this library!

Inside your python 3.11 virtual environment run:

```bash
poetry add lib-ml-REMLA10-2024
```

or install with `pip`

```bash
pip install lib-ml-REMLA10-2024
```

Now you can import the library inside python modules

```python
from lib_ml_remla import preprocess_data, split_data
```
## Tests
To run the tests run the command ```pytest``` from python3.11 virtual environemt. 

## 🛡 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/remla24-team10/lib-ml/blob/main/LICENSE) for more details. -->


# 📦 lib-ml-REMLA10-2024

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Overview

`lib-ml-REMLA10-2024` provides essential functions for preprocessing and postprocessing data in machine learning projects. It includes utilities for data splitting, preprocessing, and evaluation.

## 🛠️ Installation

> Note: Python 3.11 is required for this library!
> 

### Using Poetry

Inside your Python 3.11 virtual environment, run:

```bash
poetry add lib-ml-REMLA10-2024
```

### Using pip

Alternatively, you can install the package with pip:

```bash
pip install lib-ml-REMLA10-2024
```

## 📚 Usage

### Importing the Library

You can import the necessary functions in your Python modules:

```python
from lib_ml_remla import preprocess_data, split_data
```

###  Usage examples

### 🔄 Preprocessing Data

```python
from lib_ml_remla import preprocess_data, split_data

# Example data
train_data = ["1\tThis is a sample training sentence.", "0\tAnother training example."]
test_data = ["1\tThis is a sample test sentence."]
val_data = ["0\tThis is a sample validation sentence."]

# Split data
raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test = split_data(train_data, test_data, val_data)

# Preprocess data
X_train, y_train, X_val, y_val, X_test, y_test, char_index, tokenizer, encoder = preprocess_data(
    raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test
)
```

### 📈 Evaluating Results

```python
from lib_ml_remla import predict_classes, evaluate_results
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = load_model('path_to_your_model')

# Predict classes
labels, probabilities = predict_classes(model, encoder, X_test)

# Evaluate results
results = evaluate_results(y_test, labels)
print(results)
```

## 🛡 License

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/remla24-team10/lib-ml/blob/main/LICENSE) for more details.