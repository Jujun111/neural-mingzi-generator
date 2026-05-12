import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_MANIFEST_PATH = Path("deployment_artifacts.json")


def load_manifest(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("Artifact manifest must contain an 'artifacts' list.")
    return artifacts


def artifact_url(base_url: str, asset_name: str) -> str:
    return f"{base_url.rstrip('/')}/{asset_name}"


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with urlopen(url, timeout=120) as response:
        with temporary_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temporary_path.replace(target_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download deployment model artifacts from a release URL.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--base-url", default=os.getenv("CNG_ARTIFACT_BASE_URL", ""))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    artifacts = load_manifest(manifest_path)
    missing = [item for item in artifacts if not Path(item["path"]).exists()]

    if args.check_only:
        if missing:
            for item in missing:
                print(f"missing: {item['path']} (release asset: {item['asset']})")
            return 1
        print("All deployment artifacts are present.")
        return 0

    if not missing:
        print("All deployment artifacts are already present.")
        return 0

    if not args.base_url:
        print(
            "CNG_ARTIFACT_BASE_URL is required because deployment artifacts are missing.",
            file=sys.stderr,
        )
        for item in missing:
            print(f"missing: {item['path']} (release asset: {item['asset']})", file=sys.stderr)
        return 2

    for item in missing:
        target_path = Path(item["path"])
        url = artifact_url(args.base_url, item["asset"])
        print(f"Downloading {item['asset']} -> {target_path}")
        try:
            download_file(url, target_path)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Failed to download {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3

    print("Deployment artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
