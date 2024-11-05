FROM ghcr.io/ophoperhpo/langflow:latest-dev

CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
