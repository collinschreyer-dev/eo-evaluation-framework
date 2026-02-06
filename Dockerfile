FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY run_eval.py .
COPY EO_ICON.jpeg .

# Copy directories
COPY config/ config/
COPY prompts/ prompts/
COPY datasets/ datasets/
COPY src/ src/

# Copy existing results (including benchmark.db)
COPY results/ results/

# Create data directory for persistent storage
RUN mkdir -p /data

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start command
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
