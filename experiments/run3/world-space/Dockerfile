# The run-3 world (worlds/chatbot3) as a Docker Space: `flock world serve` (FastAPI, World API
# v1) on port 7860. Build context: the folder prepare.sh stages (this file, start.sh, the flock
# wheel, worlds/, world.yaml). Not deployed by anything in this repo.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user dist/ /app/dist/
RUN pip install --no-cache-dir "$(ls /app/dist/flock-*.whl)[controller]" "huggingface_hub==2.2.0"
COPY --chown=user worlds/ /app/worlds/
COPY --chown=user world.yaml start.sh /app/
USER user
ENV CHATBOT_CONFIG=/app/world.yaml PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["bash", "/app/start.sh"]
