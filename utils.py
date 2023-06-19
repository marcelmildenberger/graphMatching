import csv
from typing import Sequence


def read_tsv(path: str, header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        if header:
            next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[3])
    return data, uid


def save_tsv(data, path: str, delim: str = "\t"):
    with open(path, "w", newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        csvwriter.writerows(data)
