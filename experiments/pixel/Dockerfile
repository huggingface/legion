# Canvas service image. Build from experiments/pixel:
#   docker build -t flock-pixel-canvas experiments/pixel
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "numpy>=1.26" "pillow>=10"
COPY canvas ./canvas
COPY cli ./cli
COPY replay ./replay
COPY assets ./assets
ENV CANVAS_HOST=0.0.0.0 CANVAS_PORT=8765 CANVAS_DATA=/data PATH="/app/cli:${PATH}"
EXPOSE 8765
CMD ["python", "-m", "canvas"]
