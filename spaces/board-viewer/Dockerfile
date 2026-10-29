FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[controller]' 'huggingface_hub>=2.2.0'
USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-m", "flock.board_viewer.app"]
