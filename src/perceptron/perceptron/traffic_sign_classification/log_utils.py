"""Logging helper functions for experiment result files."""

import os
import csv
import re


def append_csv_row(csv_path, fieldnames, row):
    """Append one result row to a CSV file.

    Creates the file header automatically when the target file does not exist
    or is empty.

    Args:
        csv_path: Path to the CSV file.
        fieldnames: Ordered list of CSV column names.
        row: Mapping containing one row of values.

    Returns:
        None.
    """
    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_next_result_id(results_csv_path):
    """Return the next sequential run identifier from a results CSV file.

    Args:
        results_csv_path: Path to the results CSV file.

    Returns:
        Integer run id to use for the next row.
    """
    if not os.path.exists(results_csv_path) or os.path.getsize(results_csv_path) == 0:
        return 1

    max_id = 0
    with open(results_csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = str(row.get('id', '')).strip()
            match = re.match(r'^(\d+)', raw_id)
            if match:
                max_id = max(max_id, int(match.group(1)))

    return max_id + 1