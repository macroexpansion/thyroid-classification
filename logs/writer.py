from typing import List

import csv
import os


def log_writer(headers: List[str], rows: List[List[int]], foldername: str = "log", filename: str = "log.csv"):
    if not os.path.exists(os.path.join("logs", foldername)):
        os.mkdir(os.path.join("logs", foldername))

    path = os.path.join("logs", foldername, filename)
    with open(path, "w", newline="") as fo:
        writer = csv.writer(fo, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    log_writer(
        ["a", "f", "r", "d"],
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
    )
