"""
inspect_db.py  —  sample rows from corpus.db and show their token representation.

Usage:
  uv run inspect_db.py
  uv run inspect_db.py --n 20
  uv run inspect_db.py --seed 42
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.2",
#   "tqdm>=4.66",
#   "numpy>=1.26",
# ]
# ///

import argparse
import random
import sqlite3

from city_gpt import DB_PATH, SOS_TOKEN, EOS_TOKEN, FIELD_SEP, build_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and inspect corpus.db rows")
    parser.add_argument("--n",    type=int, default=10,   help="number of rows to show")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    stoi, itos = build_vocab()
    sos_id = stoi[SOS_TOKEN]
    eos_id = stoi[EOS_TOKEN]
    sep_id = stoi[FIELD_SEP]

    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT MAX(id) FROM records").fetchone()[0]
    ids   = random.sample(range(1, total + 1), min(args.n, total))

    placeholders = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT id, tokens FROM records WHERE id IN ({placeholders}) ORDER BY id",
        ids,
    ).fetchall()
    con.close()

    col_w = 6   # id column width

    for row_id, text in rows:
        # full token sequence as the model sees it: SOS + record chars + EOS
        token_ids = [sos_id] + [stoi[ch] for ch in text if ch in stoi]
        # the stored text already ends with \n (EOS); find sep position
        sep_pos  = next((i for i, t in enumerate(token_ids) if t == sep_id), None)

        print(f"{'id':>{col_w}}: {row_id}")
        print(f"{'text':>{col_w}}: {text!r}")
        print(f"{'tokens':>{col_w}}: {token_ids}")
        print(f"{'length':>{col_w}}: {len(token_ids)} tokens  "
              f"(prefix={sep_pos + 1 if sep_pos else '?'}  "
              f"name={len(token_ids) - (sep_pos + 1) if sep_pos else '?'})")

        # annotated breakdown
        parts = []
        for i, tid in enumerate(token_ids):
            ch = itos.get(tid, "?")
            if tid == sos_id:
                parts.append(f"[SOS=0]")
            elif tid == eos_id:
                parts.append(f"[EOS=1]")
            elif tid == sep_id:
                parts.append(f"[SEP={tid}]")
            else:
                parts.append(f"{ch!r}={tid}")
        print(f"{'detail':>{col_w}}: {' '.join(parts)}")
        print()


if __name__ == "__main__":
    main()
