# project/bq_utils.py
from threading import BoundedSemaphore
from google.cloud import bigquery

_MAX_BQ_JOBS = 10
_semaphore = BoundedSemaphore(_MAX_BQ_JOBS)
_client = bigquery.Client()

def run_query(sql: str):
    """
    Acquires the global BQ semaphore, fires off the job, then waits.
    """
    with _semaphore:
        job = _client.query(sql)
        return job.result()
