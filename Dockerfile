FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install -y wget tesseract-ocr libtesseract-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY . .
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "uvicorn", "app:main", "--host", "0.0.0.0", "--port", "7860"]