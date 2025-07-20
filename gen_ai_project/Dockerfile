# Use a lightweight Python image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .



# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .


# Expose a port if your application runs a server (e.g., for an API)
# EXPOSE 8000

# Command to run the application (replace with your actual entry point)
# CMD ["python", "src/main.py"]