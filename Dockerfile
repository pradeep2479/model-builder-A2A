# Start from a standard, lightweight Python base image.
# This specifies the Python version to use inside the container.
FROM python:3.11-slim

# Set the working directory inside the container.
# All subsequent commands will run from here.
WORKDIR /app

# Copy the requirements file into the container first.
# This is a Docker optimization trick.
COPY requirements.txt .

# Install all the Python dependencies listed in the file.
# --no-cache-dir makes the final image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy all of your agent scripts into the container.
COPY ./scripts/ /app/scripts/

# The container is now built. It contains Python, all libraries, and your code.
# The default command could be set here, but we will specify it at runtime.