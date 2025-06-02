# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC separately (if not in requirements)
RUN pip install --no-cache-dir dvc[gs,s3]

# Copy the entire project
COPY . .

# Optional: pull data from remote storage (if needed)
# RUN dvc pull

# Run DVC pipeline
CMD ["dvc", "repro"]
