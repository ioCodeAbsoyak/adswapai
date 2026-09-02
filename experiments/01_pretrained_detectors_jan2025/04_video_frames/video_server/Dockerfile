# AdSwapAI R&D, 2025-01-28: video server image.
FROM python:3.10-slim

WORKDIR /app

# Install the required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application
COPY app.py .
RUN mkdir videos

EXPOSE 5001

CMD ["python", "app.py"]
