"""
bq_utils.py

BigQuery concurrency control utilities.

This module centralizes all BigQuery interactions and ensures that at most
_MAX_BQ_JOBS are running at any one time by using a shared threading.BoundedSemaphore.

Public functions:
- run_query(sql: str) -> RowIterator
    * Wraps Client.query(sql) / job.result() in semaphore logic.
    * Logs a warning if the concurrency limit is reached before waiting.
- upload_dataframe(dataframe, destination_table, job_config=None) -> LoadJob
    * Wraps Client.load_table_from_dataframe(...) / job.result() in the same semaphore logic.
    * Ensures uploads and queries share the same concurrency limit.

Semaphore behavior in this module:
- _semaphore.acquire(blocking=False):
    * Attempts to grab a slot immediately.
    * If it fails (no slots), logs a warning and falls through to a blocking acquire.
- _semaphore.acquire():
    * Blocks the calling thread until a slot frees up.
- _semaphore.release():
    * Frees one slot, allowing another waiting thread to proceed.

Usage:
    from project.bq_utils import run_query, upload_dataframe
    rows = run_query("SELECT * FROM dataset.table")
    job = upload_dataframe(df, "dataset.new_table")
"""
import logging
from threading import BoundedSemaphore
from google.cloud import bigquery
import pandas as pd

logger = logging.getLogger(__name__)

_MAX_BQ_JOBS = 10
_semaphore = BoundedSemaphore(_MAX_BQ_JOBS)
_client = bigquery.Client()

def run_query(sql: str):
    """
    Params:
    - sql (str): SQL to run.

    Returns:
    - RowIterator: query results.
    """
    # try non-blocking acquire to see if we're at capacity
    got_slot = _semaphore.acquire(blocking=False)
    if not got_slot:
        logger.info(
            "BigQuery concurrency limit reached (%d). Waiting for free slot…",
            _MAX_BQ_JOBS,
        )
        _semaphore.acquire()  # now block until available

    try:
        job = _client.query(sql)
        return job.to_dataframe()
    finally:
        _semaphore.release()

def upload_dataframe(
    dataframe: pd.DataFrame,
    destination_table: str,
    job_config: bigquery.LoadJobConfig = None
) -> bigquery.LoadJob:
    """
    Params:
    - dataframe (DataFrame): data to upload.
    - destination_table (str): e.g. "dataset.table".
    - job_config (LoadJobConfig, optional): load settings.

    Returns:
    - LoadJob: the upload job (already finished).
    """
    got_slot = _semaphore.acquire(blocking=False)
    if not got_slot:
        logger.info(
            "BigQuery concurrency limit reached (%d). Waiting for free slot…",
            _MAX_BQ_JOBS,
        )
        _semaphore.acquire()

    try:
        job = _client.load_table_from_dataframe(
            dataframe, destination_table, job_config=job_config
        )
        return job.result()
    finally:
        _semaphore.release()
