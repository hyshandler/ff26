"""One-time move of the nflverse disk cache from the old user-level cache dir
(~/.cache/ff_model) into the repo's assets/nflverse_cache, now that nflverse.py
writes there directly. Safe to run multiple times: no-ops once the old dir is gone.
"""

import shutil
from pathlib import Path

import appdirs

OLD_CACHE_DIR = Path(appdirs.user_cache_dir("ff_model", "ff26"))
NEW_CACHE_DIR = Path(__file__).resolve().parents[1] / "assets" / "nflverse_cache"


def main() -> None:
    if not OLD_CACHE_DIR.exists():
        print(f"Nothing to migrate: {OLD_CACHE_DIR} does not exist.")
        return

    NEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for old_path in OLD_CACHE_DIR.rglob("*"):
        if old_path.is_dir():
            continue
        new_path = NEW_CACHE_DIR / old_path.relative_to(OLD_CACHE_DIR)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
        moved += 1

    shutil.rmtree(OLD_CACHE_DIR)
    print(f"Moved {moved} file(s) from {OLD_CACHE_DIR} to {NEW_CACHE_DIR}.")


if __name__ == "__main__":
    main()
