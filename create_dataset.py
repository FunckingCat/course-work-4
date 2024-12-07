import os
from pathlib import Path


def merge_files_with_prefixes(directory: str, file_map: dict):
    output_dir = Path(directory) / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dest_file, prefixes in file_map.items():
        dest_path = output_dir / dest_file

        with open(dest_path, "w", encoding="utf-8") as out_file:
            for file in Path(directory).iterdir():
                if file.is_file() and any(file.name.startswith(prefix) for prefix in prefixes):
                    with open(file, "r", encoding="utf-8") as in_file:
                        out_file.write(in_file.read())
                        out_file.write("\n")


if __name__ == "__main__":
    directory_path = "./resources/clean_result"
    file_mapping = {
        "medicine.txt": ["arvi", "ocd"],
        "cloud_infra.txt": ["co", "sla"],
    }

    merge_files_with_prefixes(directory_path, file_mapping)