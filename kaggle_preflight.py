"""Checks run before a Kaggle session: the mounted code matches this working
copy, and no credential file is present.
"""
from __future__ import annotations

import womd.runtime_env
import sys
from pathlib import Path

from womd import contract


def assert_working_copy_current(mounted_contract_path: Path | str) -> None:
    """Fails if the Kaggle notebook's mounted code does not match this working
    copy's staging code version.
    """
    mounted_text = Path(mounted_contract_path).read_text()
    if (f'STAGING_CODE_VERSION = "{contract.STAGING_CODE_VERSION}"'
            not in mounted_text):
        raise SystemExit(f"working copy version {contract.STAGING_CODE_VERSION}"
                         f" not found in {mounted_contract_path}")


def assert_no_credentials(working_directory: Path | str) -> None:
    """Fails if any .json file under working_directory looks like a credential
    file, so one never ends up in a saved Kaggle version.
    """
    credential_markers = ("private_key", "refresh_token")
    credential_paths = []
    for json_path in Path(working_directory).rglob("*.json"):
        json_text = json_path.read_text(errors="ignore")
        if json_path.name == "gcloud.json" or any(
                marker in json_text for marker in credential_markers):
            credential_paths.append(json_path)
    if credential_paths:
        raise SystemExit(
            f"credential files present, do NOT Save Version: {credential_paths}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "first":
        assert_working_copy_current(sys.argv[2])
    else:
        assert_no_credentials(sys.argv[2])
