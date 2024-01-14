# Code Base and Dependency

## Dependency

There are a number of dependencies to make the codebase run. These are:

- Linux is necessary. The code will not work on Windows unless you have installed tesseract locally. On linux, this can be downloaded using `apt-get install tesseract-ocr libtesseract-dev`
- Miniconda is also required. This is required because we have a dependancy of `poppler` and I could not install it via `pip`. However, if this can be installed via `pip` then we can get away with using only the python `venv`.
- For a full list of required python libraries, see the `requirements.txt` or `environment.yml` file.

> You can have a look at the `Dockerfile` to get an idea of what steps are required to make the code workable in your machine.

## Dockerfile

I have provided a `Dockerfile` to make it easy to share the code with the dependencies listed. If you have docker installed, use `docker build -t v1` to build a container with all the dependencies. After this, you can run the docker image by using `docker run -p 07860:0786 v1`.

## Code Base

The code base is somewhat large. This is because I have tried to make the project as robust as possible. We will discuss the code base module by module. Each module has comprehensive docstrings. You can have a look there for more information.

### `utilities.py`

This module has some custom exception classes, a decorator `timeout` that was useful while extracting the text from PDF URLs. The class `PDFExtractor` implements some methods to extract the textual, tabular and image information from a PDF.

### `create_dataset.py`

This module has some helper functions to create the JSON (`create_json` function) metadata from the PDF and then use this JSON metadata to create the final dataframe (using `create_dataframe` function). The JSON files created by the `create_json` function will be stored in `data/train_json` or `data/test_json` directory. The final dataframes created using the `create_dataframe` function will be stored in the `data` directory.

### `model.py`

This file has a number of classes. These are:

- `PDFDataLoader`: A class that can be used to load the data to torch model. This will be used in the `PDFDataSet` class to create the final datasets.
- `PDFDataSet`: Class for loading and preprocessing PDF classification datasets. Has methods for creating the three datasets (train, validation and test) and getting their embeddings.
- `PDFModel` A simple deep learning model with variable number of layers and number of neurons.

Apart from these, the module as methods to evaluate three models:

- `PDFModel`
- `RandomForest`
- `XGBoostClassifier`

And to train and save the final model.

### `torch_train.py`

This has a single class `TorchTrain` that I wrote way earlier to make it easy to train Pytorch models. The class has some methods to train arbitrary Pytorch model with given optimizer, loss function, metrics, scheduler, epochs etc.

### `inference.py`

Has the class `Inference` to make it easy to make inference by passing the PDF URL.

### `app.py`

Contains the web app written in gradio.
