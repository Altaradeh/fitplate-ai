FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3-pip git curl && \
    curl -sSL https://install.python-poetry.org | python3.10 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app
COPY . /app
RUN poetry install --no-interaction --no-ansi

EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "main.py"]
