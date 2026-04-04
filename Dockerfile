FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Keep build fast and deterministic for CI image validation.
COPY src ./src
COPY pyproject.toml README.md ./

CMD ["python", "-c", "print('NeuroSynth container image ready')"]
