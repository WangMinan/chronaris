"""Restore archived source checkouts without changing frozen result files."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ARCHIVE = Path(__file__).resolve().parent
REPOSITORY = ARCHIVE.parents[2]


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.PIPE)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def checksums(record):
    path = ARCHIVE / safe_relative(record["checksums"])
    if sha(path) != record["checksums_sha256"]:
        raise ValueError("Archived checksum manifest changed")
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def verify(record, destination):
    expected = checksums(record)
    if git(destination, "rev-parse", "HEAD").decode().strip() != record["commit"]:
        raise ValueError("Restored commit differs")
    tracked = set(git(destination, "ls-files", "-z").decode().split("\0")[:-1])
    if tracked != set(expected):
        raise ValueError("Restored tracked inventory differs")
    for name, item in expected.items():
        path = destination / safe_relative(name)
        if item["kind"] == "symlink":
            valid = path.is_symlink() and os.readlink(path) == item["target"]
        else:
            valid = (not path.is_symlink() and path.is_file()
                     and sha(path) == item["sha256"]
                     and bool(path.stat().st_mode & 0o111) == item["executable"])
        if not valid:
            raise ValueError(f"Restored file differs: {path}")
    for name, target in record["symlinks"].items():
        path = destination / safe_relative(name)
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError(f"Asset link differs: {path}")
        if not path.resolve().exists():
            raise ValueError(f"Asset target unavailable: {path}")
    if git(destination, "status", "--porcelain").decode().splitlines() != record["status"]:
        raise ValueError("Restored working-tree status differs")
    return len(expected)


def restore(record, root=None):
    destination = (root / record["name"]) if root else Path(record["original_path"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    checksums(record)
    patch = record["patch"]
    if patch and sha(ARCHIVE / safe_relative(patch["file"])) != patch["sha256"]:
        raise ValueError("Archived patch changed")
    for target in record["symlinks"].values():
        if not Path(target).exists():
            raise ValueError(f"Restore original assets first: {target}")
    git(REPOSITORY, "cat-file", "-e", record["commit"] + "^{commit}")
    git(REPOSITORY, "worktree", "add", "--detach", str(destination), record["commit"])
    if patch:
        git(destination, "apply", "--binary", str(ARCHIVE / safe_relative(patch["file"])))
    for name, target in record["symlinks"].items():
        path = destination / safe_relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    return dict(name=record["name"], path=str(destination), verified_files=verify(record, destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Runtime directory names from inventory.json")
    parser.add_argument("--all", action="store_true", help="Restore every archived runtime")
    parser.add_argument("--root", type=Path, help="Alternative parent for restoration testing")
    args = parser.parse_args()
    records = json.loads((ARCHIVE / "inventory.json").read_text())["worktrees"]
    by_name = {row["name"]: row for row in records}
    if args.all == bool(args.names) or len(args.names) != len(set(args.names)) or set(args.names) - by_name.keys():
        parser.error("Choose --all or distinct known runtime names")
    selected = records if args.all else [by_name[name] for name in args.names]
    print(json.dumps([restore(row, args.root.resolve() if args.root else None) for row in selected], indent=2))


if __name__ == "__main__":
    main()
