from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# the "link": go to the project, use the ML Python
ML = "cd /opt/airflow/project && /opt/ml-venv/bin/python"

with DAG(
    dag_id="smoke_test",
    start_date=datetime(2026, 11, 1),
    schedule=None,      # no schedule: runs only when you click ▶
    catchup=False,
) as dag:

    versions = BashOperator(
        task_id="versions",
        bash_command=f"{ML} -c 'import pandas, sklearn; print(pandas.__version__, sklearn.__version__)'",
    )

    tests = BashOperator(
        task_id="run_tests",
        bash_command=f"{ML} -m pytest -q",
    )

    versions >> tests