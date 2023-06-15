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

def save_csv(data, path: str):
    with open(path, "w", newline="") as f:
        csvwriter = csv.writer(f, delimiter="\t")
        csvwriter.writerows(data)
