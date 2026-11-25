# Towers as Rain Gauges: the pages and the live watch in one container (CPU only, any machine with Docker).
#   docker build -t rain-towers .
#   docker run -p 127.0.0.1:8000:8000 -v rain-towers-data:/app/data rain-towers     then open http://localhost:8000
# The first start downloads the demo data (about 300 MB); the volume keeps watched cities between runs.
FROM python:3.12-slim

# ALARM_EDIT=any: the page may change the alarm email. Inside Docker every browser looks like another machine, so
# the "only from this computer" rule would lock it forever; publish the port on 127.0.0.1 to keep it local.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000 ALARM_EDIT=any
WORKDIR /app

# libexpat: needed by the map-data library (rasterio); not in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

# CPU build of PyTorch (small); the models are tiny and run fine on CPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-demo.txt pyproject.toml ./
RUN pip install -r requirements-demo.txt
COPY . .
# /app/data is not in the image (.dockerignore), so create it owned by the app user: a new named volume
# mounted there copies this ownership. start.sh also repairs volumes made by older images, then drops root.
RUN pip install --no-deps -e . && sed -i 's/\r$//' docker/start.sh && chmod +x docker/start.sh \
 && useradd -m -u 1000 app && mkdir -p /app/data && chown -R app:app /app

EXPOSE 8000
CMD ["docker/start.sh"]
