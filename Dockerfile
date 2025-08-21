# --- Layer 1: The Foundation ---
# Every Dockerfile starts with a base image. We are choosing an official
# Python image. '3.11-slim' means we get Python 3.11 on a minimal
# version of Debian Linux, which makes our final image smaller and more secure.
FROM python:3.11-slim

# --- Layer 2: Setting up the Workspace ---
# This command sets the "working directory" inside the container.
# Think of it as creating a folder at `/app` and then running `cd /app`.
# All subsequent commands will happen from inside this directory.
WORKDIR /app

# --- Layer 3: Installing Dependencies (The Smart Way) ---
# We copy ONLY the requirements.txt file first.
# Docker builds in layers. If this file doesn't change, Docker can reuse
# the cached result of the next step, making future builds much faster.
COPY requirements.txt .

# The RUN command executes a command-line instruction inside the container
# during the build process. Here, we install all our Python libraries.
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 4: Copying Your Code ---
# Finally, we copy our application code.
# The first path `./scripts/` is on your local machine.
# The second path `/app/scripts/` is the destination inside the container.
COPY ./scripts/ /app/scripts/