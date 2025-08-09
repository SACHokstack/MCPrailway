# Use an official Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install uv
RUN pip install uv

# Install dependencies
# If you use requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Or if you use pyproject.toml
# RUN uv pip install .

# Expose port (change if needed)
EXPOSE 8080

# Start your app
CMD ["uv", "run", "python", "mcp_starter.py"]
