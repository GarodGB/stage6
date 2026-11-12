from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

ML = "cd /opt/airflow/project && /opt/ml-venv/bin/python"

with DAG(
    dag_id="retrain",
    start_date=datetime(2026, 11, 1),
    schedule="@monthly",          # for now; we'll decide the real policy in Task 9
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    train = BashOperator(task_id="train", bash_command=f"{ML} -m train.train")
    test = BashOperator(task_id="test", bash_command=f"{ML} -m inference.test")

    train >> test