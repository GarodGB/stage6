# ---- Block 1: start from Airflow (already installed, Python 3.12) ----
FROM apache/airflow:2.9.3-python3.12

# ---- Block 2: create the separate ML Python at /opt/ml-venv ----
USER root
RUN python -m venv /opt/ml-venv && chown -R airflow:0 /opt/ml-venv
USER airflow

# ---- Block 3: install YOUR libraries into the ML Python only ----
COPY requirements.txt /tmp/requirements.txt
RUN PIP_USER=false /opt/ml-venv/bin/pip install --no-cache-dir --retries 10 --timeout 120 -r /tmp/requirements.txt

# ---- Block 4: copy your project code in (last, because it changes most) ----
COPY --chown=airflow:0 . /opt/airflow/project