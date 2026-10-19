"""Runs at the end of each Kaggle session and fails if any file in the working
directory looks like a credential, so one never ends up in a saved Kaggle
version. Usage: kaggle_preflight.py WORKING_DIRECTORY.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Text that appears in a Google service-account key or an OAuth token.
CREDENTIAL_MARKERS = ("private_key", "refresh_token")


def is_credential(json_path: Path) -> bool:
    if json_path.name == "gcloud.json":
        return True
    text = json_path.read_text(errors="ignore")
    return any(marker in text for marker in CREDENTIAL_MARKERS)


def main() -> None:
    working_directory = Path(sys.argv[1])
    json_paths = working_directory.rglob("*.json")
    credential_paths = [path for path in json_paths if is_credential(path)]
    if credential_paths:
        raise SystemExit(
            f"credential files present, do NOT Save Version: {credential_paths}"
        )


if __name__ == "__main__":
    main()
