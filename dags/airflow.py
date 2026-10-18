# Import necessary libraries and modules
from airflow import DAG
# from airflow.operators.python import PythonOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.sdk.exceptions import AirflowFailException
from datetime import datetime, timedelta
from src.lab import load_data, validate_data, data_preprocessing, build_save_model, load_model_elbow

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

# Define default arguments for your DAG
default_args = {
    'owner': 'Connor Rivera',
    'start_date': datetime(2026, 1, 15),
    'retries': 2,  # Retry a failed task twice before marking it failed
    'retry_delay': timedelta(minutes=1),  # Wait 1 minute between retries
    'execution_timeout': timedelta(minutes=10),  # Fail any task that runs longer than 10 minutes
}

def choose_path(problems):
    """Branch callable: train if validation found no problems, otherwise report the bad data."""
    if problems:
        return 'report_bad_data_task'
    return 'data_preprocessing_task'


def report_bad_data(problems):
    """Fail the run on purpose and list every validation problem."""
    for problem in problems:
        print(f"Validation problem: {problem}")
    raise AirflowFailException(f"Data failed validation with {len(problems)} problem(s); model was not trained")

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='K-Means clustering pipeline with data validation (DADS 7305 Lab 2)',
    schedule='0 6 * * 1',  # Cron schedule: every Monday at 06:00 UTC
    catchup=False,
    tags=['dads7305', 'lab2', 'clustering'],
    params={'data_file': 'file.csv'},  # Which CSV in dags/data to load; change it when triggering
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
        op_kwargs={'filename': '{{ params.data_file }}'},
    )

    # Task to check the loaded data for problems, depends on 'load_data_task'
    validate_data_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=validate_data,
        op_args=[load_data_task.output],
    )

    # Branch: choose the training path or the bad-data path based on validation
    check_validation_task = BranchPythonOperator(
        task_id='check_validation_task',
        python_callable=choose_path,
        op_args=[validate_data_task.output],
    )

    # Task to report validation problems and fail the run, runs only on the bad-data path
    report_bad_data_task = PythonOperator(
        task_id='report_bad_data_task',
        python_callable=report_bad_data,
        op_args=[validate_data_task.output],
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    # Set task dependencies
    load_data_task >> validate_data_task >> check_validation_task
    check_validation_task >> [data_preprocessing_task, report_bad_data_task]
    data_preprocessing_task >> build_save_model_task >> load_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
