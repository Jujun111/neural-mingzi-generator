import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db_loader import get_cbdb_names
from markov_chain import ChineseNameMarkov


def build_markov_artifact(db_path: Path, output_path: Path) -> None:
    surnames, given_names = get_cbdb_names(db_path=str(db_path))
    if not surnames or not given_names:
        raise RuntimeError(f"No usable names were loaded from {db_path}.")

    model = ChineseNameMarkov(surnames_list=surnames, boost_compound=False)
    model.train(given_names)

    with output_path.open("wb") as handle:
        pickle.dump(model, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a serialized Markov artifact from CBDB.")
    parser.add_argument("--db-path", default="latest.db", help="Path to the CBDB SQLite database.")
    parser.add_argument(
        "--output-path",
        default="markov_model.pkl",
        help="Where to write the serialized Markov artifact.",
    )
    args = parser.parse_args()

    build_markov_artifact(Path(args.db_path), Path(args.output_path))
    print(f"Markov artifact written to {args.output_path}")


if __name__ == "__main__":
    main()
