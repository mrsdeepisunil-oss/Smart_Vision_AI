FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code into the container
COPY . .

# Expose the default port Hugging Face looks for
EXPOSE 7860

# --- FIX IS HERE: Force Streamlit to use port 7860 ---
CMD ["streamlit", "run", "mainapp.py", "--server.port=7860", "--server.address=0.0.0.0"]