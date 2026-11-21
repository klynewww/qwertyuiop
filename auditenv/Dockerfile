FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY LICENSE Makefile pytest.ini judge_adapter.py ./
COPY env/ env/
COPY policies/ policies/
COPY evaluator/ evaluator/
COPY qa/ qa/
COPY generator/ generator/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/

RUN useradd --create-home audit && chown -R audit /app
USER audit

# сеть не нужна: запускать с  docker run --network none
CMD ["make", "check", "audit"]
