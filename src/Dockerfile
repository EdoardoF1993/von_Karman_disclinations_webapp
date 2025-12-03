# ------------------------------------------------------------
# 1. Start from the official DOLFINx image (includes PETSc, MPI,
#    Basix, UFL, FFCx, Dolfinx Python packages, and all deps)
# ------------------------------------------------------------
FROM dolfinx/dolfinx:v0.8.0

# ------------------------------------------------------------
# 2. System dependencies (optional but useful)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# 3. Create working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# 4. Copy only requirements first (to take advantage of Docker caching)
# ------------------------------------------------------------
COPY requirements.txt .

# ------------------------------------------------------------
# 5. Install Python dependencies
#
# IMPORTANT:
#  - Do NOT install fenics-dolfinx, fenics-basix, fenics-ffcx manually
#    → these are already inside dolfinx/dolfinx:stable
#  - Remove them from requirements.txt or ignore them during pip install
# ------------------------------------------------------------

# Remove fenics packages from requirements to avoid pip errors
RUN sed -i '/fenics-/d' requirements.txt

# Install the rest
RUN pip3 install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 6. Copy the entire project
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# 7. Set the Render web port (Flask must bind to 0.0.0.0:$PORT)
# ------------------------------------------------------------
ENV PORT=10000

# ------------------------------------------------------------
# 8. Set default command — run your Flask app
# ------------------------------------------------------------
CMD ["python3", "app.py"]
