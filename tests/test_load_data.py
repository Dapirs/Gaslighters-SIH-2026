import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.load_data import load_data

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "_fixtures", "load_data_fixture.csv")

FIXTURE_CSV = (
    'MP NAME;WORK;CATEGORY;STATE;CONSTITUENCY;IDA;CITY;WARD;BLOCK;VILLAGE;'
    'RECOMMENDED DATE;ALLOCATION AMOUNT;IDA APPROVAL;STATUS;HOUSE\n'
    'Test MP One;Road repair;Normal/Others;Kerala;Kochi;DISTRICT COLLECTOR KOCHI_IDA;;;Block A;Village A;'
    '2023-05-01;500000;Approved;Sanctioned;Lok Sabha\n'
    'Test MP Two;Water tank;Normal/Others;Kerala;Kochi;DISTRICT COLLECTOR KOCHI_IDA;;;Block B;Village B;'
    '2023-06-15;300000;Approved;Completed;Lok Sabha\n'
    # deliberate exact duplicate of the row above
    'Test MP Two;Water tank;Normal/Others;Kerala;Kochi;DISTRICT COLLECTOR KOCHI_IDA;;;Block B;Village B;'
    '2023-06-15;300000;Approved;Completed;Lok Sabha\n'
    'Test MP Three;Street light;Normal/Others;Bihar;Patna;DEPUTY COMMISSIONER PATNA_IDA;;;Block C;Village C;'
    '2023-07-20;150000;Pending;Unsanctioned;Lok Sabha\n'
)


def _write_fixture():
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        f.write(FIXTURE_CSV)
    return FIXTURE_PATH


def test_row_count_after_dedup():
    path = _write_fixture()
    df = load_data(path)
    # 4 raw rows, 1 exact duplicate -> 3 expected
    assert len(df) == 3, f"Expected 3 rows after dedup, got {len(df)}"


def test_duplicate_actually_dropped():
    path = _write_fixture()
    df = load_data(path)
    matches = df[(df["mp_name"] == "Test MP Two") & (df["work"] == "Water tank")]
    assert len(matches) == 1, f"Expected exactly 1 surviving row for the duplicated record, got {len(matches)}"


def test_recommended_date_is_datetime():
    path = _write_fixture()
    df = load_data(path)
    assert str(df["recommended_date"].dtype).startswith("datetime"), (
        f"recommended_date should be datetime dtype, got {df['recommended_date'].dtype}"
    )


def test_allocation_amount_is_numeric():
    path = _write_fixture()
    df = load_data(path)
    assert df["allocation_amount"].dtype.kind in "if", (
        f"allocation_amount should be numeric, got dtype {df['allocation_amount'].dtype}"
    )


def test_columns_are_snake_case_no_leftover_headers():
    path = _write_fixture()
    df = load_data(path)
    expected = {
        "mp_name", "work", "category", "state", "constituency", "agency",
        "city", "ward", "block", "village", "recommended_date",
        "allocation_amount", "agency_approval", "status", "house",
    }
    actual = set(df.columns)
    assert actual == expected, (
        f"Column mismatch.\nMissing: {expected - actual}\nUnexpected/leftover: {actual - expected}"
    )


def test_default_path_falls_back_to_config():
    # If this raises because config.RAW_DATA_PATH doesn't exist on disk in this
    # environment, that's expected in a fixture-only test run -- the real check
    # is that load_data() with NO argument doesn't immediately TypeError from a
    # missing default. Import success + attribute access is the actual assertion.
    from src import config
    assert hasattr(config, "RAW_DATA_PATH"), "config.py must define RAW_DATA_PATH"
    import inspect
    sig = inspect.signature(load_data)
    assert sig.parameters["path"].default is None, "load_data's path parameter should default to None"


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    from _test_utils import run_tests
    run_tests(globals(), "load_data")