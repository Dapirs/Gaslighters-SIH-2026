"""src/load_data.py — load and clean the raw MPLADS dataset."""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Exact column order as they appear in the raw CSV (15 columns)
_RAW_COLUMNS = [
    "mp_name",
    "work",
    "category",
    "state",
    "constituency",
    "agency",
    "city",
    "ward",
    "block",
    "village",
    "recommended_date",
    "allocation_amount",
    "agency_approval",
    "status",
    "house",
]


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the MPLADS CSV and return a clean DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.  Defaults to ``config.RAW_DATA_PATH`` when
        *None* is supplied.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with snake_case column names, ``recommended_date``
        parsed as ``datetime64[ns]``, and exact duplicate rows removed.
    """
    if path is None:
        path = config.RAW_DATA_PATH

    logger.info("Reading raw data from %s", path)
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )

    # Rename columns to snake_case in declared order
    df.columns = _RAW_COLUMNS[: len(df.columns)]

    # Parse recommended_date
    df["recommended_date"] = pd.to_datetime(
        df["recommended_date"], dayfirst=True, errors="coerce"
    )

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    logger.info(
        "Dropped %d exact duplicate rows (%d → %d rows)", dropped, before, len(df)
    )

    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df
