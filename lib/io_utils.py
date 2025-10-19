import os, yaml, pandas as pd
from pathlib import Path

def load_settings(path="settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*ps):
    for p in ps: Path(p).mkdir(parents=True, exist_ok=True)

def read_any(path_or_dir: Path) -> pd.DataFrame:
    p = Path(path_or_dir)
    # If path points to a single file, read it directly.
    if p.is_file():
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # If path is a directory, search recursively for parquet or csv files.
    if p.is_dir():
        # Prefer parquet files (including parquet dataset directories where files may live in subfolders)
        parquet_files = list(p.rglob("*.parquet"))
        if parquet_files:
            try:
                dfs = []
                for x in parquet_files:
                    d = pd.read_parquet(x)
                    # preserve source file stem so callers can recover per-file ids
                    d["__source"] = x.stem
                    dfs.append(d)
                return pd.concat(dfs, ignore_index=True)
            except ImportError as ie:
                # Common case: parquet engine not installed
                raise ImportError(
                    "Parquet files found but no parquet engine is available. "
                    "Install 'pyarrow' or 'fastparquet' in your environment (e.g. `pip install pyarrow`)."
                ) from ie
            except Exception as e:
                # If parquet files cannot be read for another reason, try CSV fallback below
                parquet_err = e

        csv_files = list(p.rglob("*.csv"))
        if csv_files:
            dfs = []
            for x in csv_files:
                d = pd.read_csv(x)
                d["__source"] = x.stem
                dfs.append(d)
            return pd.concat(dfs, ignore_index=True)

        # If parquet files existed but failed for a non-ImportError, surface that error
        if 'parquet_err' in locals():
            raise parquet_err

        # No candidate files found in directory
        raise FileNotFoundError(f"No .parquet or .csv files found under directory: {p}")

    # Path does not exist
    raise FileNotFoundError(p)