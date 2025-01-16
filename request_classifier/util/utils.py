import shutil

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import os
from loguru import logger

def download_repo_files(repo_id: str, subdir: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)

    # List all files in the repository
    all_files = list_repo_files(repo_id=repo_id, repo_type="model")

    # Filter the files to only those in the specified subdirectory
    files_to_download = [f for f in all_files if f.startswith(subdir)]

    for filename in files_to_download:
        # Determine local path
        relative_path = filename[len(subdir):]  # strip the subdir prefix
        local_subpath = os.path.join(local_dir, relative_path)

        # Check if local file already exists
        if os.path.exists(local_subpath):
            logger.info(f"Skipping {filename} because it already exists locally.")
            continue

        # Make sure nested directories exist
        os.makedirs(os.path.dirname(local_subpath), exist_ok=True)

        # Download the file (hf_hub_download returns a path in cache)
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision="main",
        )

        # Move from cache to your desired directory
        os.replace(downloaded_file_path, local_subpath)
        logger.info(f"Downloaded and saved {filename} to {local_subpath}")
