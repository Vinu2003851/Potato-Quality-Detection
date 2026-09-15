# Start from a slim official Python image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers output (cleaner logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system libraries OpenCV/Pillow/ultralytics need for image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (Docker caches this layer if requirements
# don't change, so rebuilds are faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app (code + model weights)
COPY . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
# If deploying elsewhere (e.g. your own server), 8000 is also fine —
# just make sure it matches whatever you expose/run with.
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
