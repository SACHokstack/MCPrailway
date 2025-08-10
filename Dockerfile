# Use an official Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install OS packages required for building/installing dependencies and runtime libs
# - git: for pip VCS installs
# - build-essential: compile native wheels when needed
# - libgl1, libglib2.0-0: OpenCV runtime
# - libsndfile1: soundfile/audio
# - libmagic1: python-magic
# - libgomp1: OpenMP runtime (onnxruntime, numpy, etc.)
# - ffmpeg: pydub/av related tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 libsndfile1 libmagic1 libgomp1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*


# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.deploy.txt


# Or if you use pyproject.toml
# RUN uv pip install .

# Expose port (change if needed)
EXPOSE 8000

# Start your app
CMD ["python", "mcp_starter.py"]
