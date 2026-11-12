from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

ML = "cd /opt/airflow/project && /opt/ml-venv/bin/python"

with DAG(
    dag_id="forecast",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1, 
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
) as dag:

    
    clean = BashOperator(task_id="clean", bash_command=f"{ML} -m src.clean")
    check_data = BashOperator(task_id="check_data", bash_command=f"{ML} -m src.forecast_checks check_data")
    fill_actuals = BashOperator(task_id="fill_actuals", bash_command=f"{ML} -m src.db fill_actuals")
    predict = BashOperator(task_id="predict", bash_command=f"{ML} -m inference.predict")
    check_output = BashOperator(task_id="check_output",
                                bash_command=f"{ML} -m src.forecast_checks check_output")
    update_checkpoint = BashOperator(task_id="update_checkpoint",
                                     bash_command=f"{ML} -m src.forecast_checks update_checkpoint")

    clean >> check_data >> fill_actuals >> predict >> check_output >> update_checkpoint

    

    