import gradio as gr
from inference import Inference


def predict_url_class(url):
    """Predicts the class of the given pdf url. Creates the output necessary for gradio Label."""
    inference = Inference(pdf_url=url)
    try:
        outputs = inference.predict()
    except Exception as e:
        gr.Warning(e)
    output_for_gradio = {
        "Lighting": outputs[1],
        "Non-Lighting": outputs[0],
    }
    return output_for_gradio


def main():
    # Define Gradio interface
    description = "<p>The model in trained on a number of PDFs related to lighting and non-lighting products. The model takes an URL as input and predicts whether the product in the PDF corresponds to a Ligthing product or not. The model may take upto 30 second to make a prediction. This is because we need to first extract textual, tabular and image information from various pages of the PDF and this may a long time. Make sure that the URL provided is unblocked and can be downloaded without any extra steps.</p>"
    inputs = gr.Text(lines=1, placeholder="Enter the url of the PDF", label="URL")
    outputs = gr.Label(
        num_top_classes=2,
        label="Prediction",
        every=2,
    )
    gradio_app = gr.Interface(
        fn=predict_url_class,
        inputs=inputs,
        outputs=outputs,
        title="Lighting Product Identifier",
        description=description,
        theme="snehilsanyal/scikit-learn",
        examples=[
            [
                "https://www.topbrasslighting.com/wp-content/uploads/TopBrass-138.01-tearsheet-Jun12018.pdf"
            ],
            ["https://lyntec.com/wp-content/uploads/2018/12/LynTec-XPC-Brochure.pdf"],
        ],
        allow_flagging="never",
    )
    gradio_app.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    # Run Gradio app
    main()
