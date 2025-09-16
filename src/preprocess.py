import os
import argparse
from pathlib import Path
from tqdm import tqdm

from .data import preprocess_folder
from .utils import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="Preprocess MRI dataset (skull-strip + organize)")
    parser.add_argument("--raw", default="data/raw", help="Input raw data directory (expects class subfolders)")
    parser.add_argument("--out", default="data/processed", help="Output processed data directory")
    parser.add_argument("--no_skull_strip", action="store_true", help="Disable skull-stripping heuristic")
    args = parser.parse_args()

    raw_dir = args.raw
    out_dir = args.out
    do_skull = not args.no_skull_strip

    print(f"Raw dir: {raw_dir}")
    print(f"Processed dir: {out_dir}")
    print(f"Skull-stripping: {do_skull}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    preprocess_folder(raw_dir=raw_dir, processed_dir=out_dir, do_skull_strip=do_skull)

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
