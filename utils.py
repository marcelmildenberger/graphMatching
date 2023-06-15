import csv
from typing import Sequence
def read_csv(path: str, header: bool = True) -> Sequence[Sequence[str]]:
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        if header:
            next(reader)
        for row in reader:
            data.append(row)
    return data