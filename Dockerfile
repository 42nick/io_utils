FROM python:3.8-slim-buster

WORKDIR /app

COPY src src
COPY pyproject.toml .
COPY README.md .

RUN pip install .

