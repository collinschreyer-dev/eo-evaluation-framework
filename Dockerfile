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

# Copy existing results (including benchmark.db and JSON files)
COPY results/ results/

# Create data directory for persistent storage
RUN mkdir -p /data

# Create startup script that migrates bundled data to persistent storage
RUN echo '#!/bin/bash\n\
# Migrate bundled results to persistent storage if not already done\n\
if [ -d "/data" ] && [ -w "/data" ]; then\n\
  if [ ! -f "/data/benchmark.db" ] && [ -f "/app/results/benchmark.db" ]; then\n\
    echo "Migrating bundled database to persistent storage..."\n\
    cp /app/results/benchmark.db /data/benchmark.db\n\
    cp /app/results/*.json /data/ 2>/dev/null || true\n\
    cp /app/results/*.csv /data/ 2>/dev/null || true\n\
    echo "Migration complete!"\n\
  fi\n\
fi\n\
exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start command with migration
CMD ["/app/start.sh"]
