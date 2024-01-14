# Lighting Product Prediction

## Problem Statement

We are provided with a web URLs of a number of PDFs that has description about a product. The goal is to create a model and pipeline that can take the web URL and predict whether the PDF corresponding to the given URL is a lighting product or not.

> **Note:** I have used a number of tools to solve the problem. The complete information about the code base and the tools used can be found in the file [code_and_dependancy](code_and_dependancy.md)

## Steps Involved

The main steps involved in the assignments are:

- Extracting the textual, tabular and image data from the PDF
- Creating an embedding based on the above data
- Training the model using the embedding

## Data Extraction

Each PDF, once downloaded, has a number information that can be extracted. However, only some information is extracted. These are:

- **Textual Information:** These are simple text that can be _copied_ from the PDF. I have used `pdfminer` for this.
- **Tabular Information:** These are the tables that are available in the PDF. Some of these tables might contain information regarding whether the product is a lighting product or not. For this, I used `pdfplumber`
- **Image Information:** These are the images in the PDF. These images can have some textual information and we need to extract those too. This is especially necessary in the scenario when all the pages in the PDF are images only. To extract the images, I used `pytesseract` that uses `Tesseract` from Google.

The final text content that will be used for the model training has the following format:

```text
PAGE <i>
<TEXT CONTENT>
IMAGE <i>
<IMAGE TEXT>
IMAGE <i> ENDS
TABLE <i>
<TABLE TEXT>
TABLE <i> ENDS
PAGE <i> ENDS
```

Here, `<i>` starts from 0 and varies as the number of pages, images and table changes between page to page and PDF to PDF. The idea behind using this start and end identifier is that the embedding model will, hopefully, understand what these identifiers mean and figure out some relationship. For example, the model might figure it out that the first page is the most important while determining the class of the product.

> See the class `PDFExtractor` in `utilities.py` module for more detail on how the extraction is happening.

## Embedding Model

I have used the `paraphrase-MiniLM-L3-v2` variant of the [Sentence Transformer](https://www.sbert.net/) as the embedding model. The model generates an embedding in 384 dimensional vector space. This means that the model creates a function that takes a sentence of variable length and returns a vector in $\mathbb{R}^{384}$.

$$
\Phi(\text{sentence}) = \mathbf{v}
$$

This embedding $\mathbf{v}$ is used directly in the models.

## A Note About the Dataset

Before I give detail about the training process, there are some tidbits about the dataset that I need to discuss.

### The Dataset Information

- There were two datasets, the training dataset had about 1000 items while the testing set has 80 rows.
- Some items from the training set were broken. While trying to download the PDFs using `requests`, I was getting a number of 400 level errors. These targets (about 150) have been ignored. For testing set, all the urls were valid.

### The Mismatch Between Train and Test Set

There was a lot of mismatches in the distribution of the test data and the train data. The products that were in the testing set were from the providers that were not in the training set. Due to this, the models that I trained performed well on training and validation (which was 20% of the training set) set, but the performance on the test set was not that great. One interesting things is that when I trained the model by using only 8 of the testing items, the model generalized very well. More will be dicussed in the section.

## Training the Models

### Using 10% of Testing Dataset

During the modeling experiments, I trained on two types of dataset:

- Only using the training dataset to train and validate. I will label this as type I dataset.
- Using the training dataset with 10% of testing dataset to train and validate. (The final accuracy of the model was determined by running the final model on the 90% of the remaining training dataset. This means that by all means, the model has not seen those 90% of the dataset on which the model is evaluated.). This will be labeled as type II dataset.

### Loss and Metrics

The training dataset is completely balanced but in the test set is not balanced. Since the training set is balanced and I have to use this for training purposes, I used accuracy as the metric. I also kept track of precision.

### Neural Networks

I started by creating a simple one and two layered neural network trained on the embedding. Furthermore, I experimented with the number of layers and number of neurons in the layers. The model performed very well with accuracy of 95%+ on both the training and validation set for both the type I and type II dataset. On the testing dataset, however, the model performed about 70% on the type I dataset but again, 95%+ on the type II dataset. This shows how much mismatch there is in the distribution of the training and testing set.

### Machine Learning Models

It turned out that we do not need to go all the way up to neural networks for a good performance. The good old random forest is more than enough to handle our use case. I experimented with two models: `RandomForestClassifier` and `XGBoostClassifier`. I used `optuna` to tune for the best hyperparameters. Interestingly, `RandomForestClassifier` beat `XGBoostClassifier`. So the final model that I used was `RandomForestClassifier`. The model had an accuracy of 100% on the training set and about

## How to Verify My Results?

To easily run and experiment with various models, I have created the `model.py` module. The module has all the necessary class and functions to both evaluate and train the final model. To evaluate, make sure that all the requirements are satisfied and call `model.py` with flag `--task` set to `evaluate`. That is, use:

```bash
python model.py --task evaluate
```

You can also pass a flag `--fraction` that controls what fraction of testing data you want to have in the training dataset to create the type II dataset. When running, you will get something like this:

```bash
...
Evaluating DL Model
------------------------------------------------------------
Evaluating for the model: DL Model for test dataset...
Accuracy: 0.9444444444444444
Precision: 0.8181818181818182
...
Evaluating for the model: RandomForest for train dataset...
Accuracy: 1.0
Precision: 1.0
...
Evaluating for the model: RandomForest for validation dataset...
Accuracy: 0.9732824427480916
Precision: 0.9565217391304348
...
Evaluating for the model: RandomForest for test dataset...
Accuracy: 1.0
Precision: 1.0
...
Evaluating for the model: XGBoost for train dataset...
Accuracy: 1.0
Precision: 1.0
...
Evaluating for the model: XGBoost for validation dataset...
Accuracy: 0.9465648854961832
Precision: 0.9280575539568345
...
Evaluating for the model: XGBoost for test dataset...
Accuracy: 0.8611111111111112
Precision: 0.6428571428571429
```

You can run the same module with flag `--task` set to `train` to train and save the final model. Pass `--model_save_path` flag to change the name of the file you want to save the model as. You should see something like:

```bash
...
Evaluating for the model: Final Model for train dataset...
Accuracy: 1.0
Precision: 1.0
...
Evaluating for the model: Final Model for test dataset...
Accuracy: 1.0
Precision: 1.0
```

## How to Make an Inference?

To allow for an easy inference, I have created `Inference` class in the module `inference.py`. The class takes the url of the PDF and the path of the saved model, extract the text information using the `PDFExtractor`, creates embedding using `sentences_to_embedding` method of `PDFDataSet` class and then makes a prediction. The final method `predict` returns a list of two elements which are the probabilities that the given product is a non-lighting product and lighting product.

The module can also be called via command line. Just use:

```bash
python inference.py --url https://www.waterfallaudio.com/wp-content/uploads/2017/08/Data_Sheet_Waterfall_SAT150.pdf
```

You might see something like:

```bash
This is a Non-Lighting product with a probability of 66.31%
[0.66311301 0.33688699]
```

You can also use the `--model_path` flag to pass the file path of the save model if you need to check your output on mutilple models.

## The Web App

To make the inference even easier, I have created a web app using gradio. The app sits in the module `app.py` and can be accessed locally using:

```bash
gradio app.py
```

and then going to the local [host port 7860](http://127.0.0.1:7860/). The app is also hosted on HuggingFace. You can find it [here](https://huggingface.co/spaces/hari31416/LightingProduct).

## Things That Can Be Improved

A number of things can be done to make the model better. Some are:

- Make the text extraction more robust. Extract the text formatting along with the text. Keep the same order of the text, tables and images.
- Search for better model that can generalize even better.
- Use more examples to fine tune the embedding model.
- Do a more exhaustive hyperparameter tuning.

> The model is deployed on Huggingface as a web app using Gradio. You can find the app [here](https://huggingface.co/spaces/hari31416/LightingProduct).
