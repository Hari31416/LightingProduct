from utilities import (
    PDFExtractor,
    get_simple_logger,
    ModuleException,
)
from create_dataset import clean_text
from model import PDFDataSet
import pickle
import argparse


class Inference:
    """A class that does inference given a url of the pdf. Uses the saved RandomForest model for inference."""

    def __init__(self, pdf_url, model_path="final_model.pkl") -> None:
        self.pdf_url = pdf_url
        self.model_path = model_path
        self.logger = get_simple_logger("inference", level="debug")

    def _extract_text(self):
        self.logger.debug("Extracting text from the PDF.")
        pdf_extractor = PDFExtractor(
            self.pdf_url,
            min_characters=5,
            maximum_pages=3,
        )
        try:
            final = pdf_extractor.extract_pages()
            data = {
                "status": "ok",
                "id": "001",
                "page_contents": pdf_extractor.page_contents,
                "final_content": clean_text(final),
                "url": self.pdf_url,
            }
        except ModuleException as e:
            self.logger.warning("A module exception is raised.")
            self.logger.error(e)
            raise e
        except Exception as e:
            self.logger.error(e)
            raise e
        return data, final

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        self.model = model
        return self.model

    def _create_embedding(self, text):
        dataset = PDFDataSet()
        embedding = dataset.sentences_to_embedding([text])
        return embedding.reshape(1, -1)

    def _pretty_print_probability(self, p):
        non_lighting_probability = p[0] * 100
        lighting_probablity = p[1] * 100
        if lighting_probablity > non_lighting_probability:
            print(
                f"This is a Lighting product with a probability of {lighting_probablity:.2f}%"
            )
        else:
            print(
                f"This is a Non-Lighting product with a probability of {non_lighting_probability:.2f}%"
            )

    def predict(self):
        _, sentence = self._extract_text()
        embedding = self._create_embedding(sentence)
        model = self._load_model()
        prediction = model.predict_proba(embedding)
        prediction = prediction[0]
        self._pretty_print_probability(prediction)
        return prediction


def main(args):
    inference = Inference(args.url)
    prediction = inference.predict()
    print(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()
    main(args)
