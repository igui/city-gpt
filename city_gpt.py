"""
City GPT
========
A character-level Transformer trained on the GeoNames allCountries dataset.

Architecture is a direct port of Andrej Karpathy's GPT:
  https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py

Corpus format (one record per DB row)
--------------------------------------
  <flag emoji><feature-class emoji><feature-code ascii>|<name>\\n

Examples:
  🇺🇸🏙️PPL|new york city
  🇬🇧⛰️MT|ben nevis
  🇯🇵🏙️PPLC|tokyo

Country codes  → Unicode regional-indicator flag pair (🇺🇸, 🇬🇧, …)
Feature class  → single emoji (9 classes, fixed vocab)
Feature code   → raw ASCII uppercase string + "|" separator
Name           → normalized lowercase ASCII

Storage
-------
prepare writes data/corpus.db (SQLite).  No file is loaded into memory
during training; batches are fetched on-demand with random DB queries.

Usage (via uv):
  uv run city_gpt.py prepare                      # download + build DB
  uv run city_gpt.py prepare --max-rows 500000    # smaller subset
  uv run city_gpt.py train                        # train the model
  uv run city_gpt.py generate                     # free-form generation
  uv run city_gpt.py generate --prompt "🇺🇸🏙️PPL|new"
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
import os
import random
import sqlite3
import sys
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# Hyperparameters  (identical to Karpathy's gpt.py defaults)
# ---------------------------------------------------------------------------
batch_size    = 64    # sequences processed in parallel
block_size    = 256   # maximum context length (codepoints)
max_iters     = 5000
eval_interval = 100
learning_rate = 3e-4
device        = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters    = 50
eval_samples  = 10   # number of generated records printed at each eval
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.2
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = "data"
ZIP_PATH   = os.path.join(DATA_DIR, "allCountries.zip")
RAW_PATH   = os.path.join(DATA_DIR, "allCountries.txt")
DB_PATH    = os.path.join(DATA_DIR, "corpus.db")
MODEL_PATH = "city_gpt.pt"

DATA_URL = "https://download.geonames.org/export/dump/allCountries.zip"

# GeoNames column indices (0-based after split on tab)
COL_NAME          = 1
COL_ASCIINAME     = 2
COL_FEATURE_CLASS = 6
COL_FEATURE_CODE  = 7
COL_COUNTRY_CODE  = 8

VAL_FRACTION = 0.1   # fraction of row IDs held out for validation at train time
# ---------------------------------------------------------------------------


# ===========================================================================
# Emoji / record encoding
# ===========================================================================

_RI_BASE = 0x1F1E6 - ord("A")   # regional indicator 'A'

def country_to_flag(cc: str) -> str:
    """'US' -> '🇺🇸'  deterministic, no lookup table."""
    if len(cc) != 2 or not cc.isalpha():
        return "🏳️"
    return "".join(chr(_RI_BASE + ord(c)) for c in cc.upper())


FEATURE_CLASS_EMOJI: dict[str, str] = {
    "A": "🏛️",
    "H": "💧",
    "L": "🌿",
    "P": "🏙️",
    "R": "🛣️",
    "S": "🏗️",
    "T": "⛰️",
    "U": "🌊",
    "V": "🌲",
}
_CLASS_FALLBACK = "❓"
FIELD_SEP = "|"


_KEEP_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789 '-")

def _normalize_name(raw: str) -> str:
    return "".join(ch for ch in raw.lower() if ch in _KEEP_CHARS).strip()


def _record_text(country_code: str, feature_cls: str,
                 feature_code: str, name: str) -> str:
    flag      = country_to_flag(country_code)
    cls_emoji = FEATURE_CLASS_EMOJI.get(feature_cls, _CLASS_FALLBACK)
    return f"{flag}{cls_emoji}{feature_code}{FIELD_SEP}{name}\n"


# ===========================================================================
# Vocabulary  (built statically from known character sets)
# ===========================================================================

SOS_TOKEN = "<SOS>"   # start-of-sequence  (id = 0, always)
EOS_TOKEN = "\n"      # end-of-sequence    (id = 1, always)

def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """
    Construct the full vocabulary without reading the corpus.

    Token layout:
      0  <SOS>  start-of-sequence (generation seed)
      1  \\n     end-of-sequence / record terminator
      2+ all other characters, sorted

    Characters that can appear in a record:
      - Regional indicator codepoints for all A-Z pairs (flag emojis)
      - Feature-class emojis (fixed set)
      - '❓' fallback / '🏳️' unknown flag pieces
      - ASCII uppercase A-Z + digits 0-9 (feature codes)
      - '|'  field separator
      - name chars: a-z 0-9 space ' -
    """
    chars: set[str] = set()

    # regional indicator codepoints  U+1F1E6 .. U+1F1FF  (26 letters)
    for i in range(26):
        chars.add(chr(0x1F1E6 + i))

    # feature-class emojis
    for emoji in FEATURE_CLASS_EMOJI.values():
        chars.update(emoji)
    chars.update(_CLASS_FALLBACK)
    chars.update("🏳️")

    # ASCII uppercase for feature codes
    chars.update("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # field separator and name characters (no \n — handled as EOS below)
    chars.add(FIELD_SEP)
    chars.update(_KEEP_CHARS)

    # SOS=0, EOS(\n)=1, then all other chars sorted from id=2 onward
    vocab = [SOS_TOKEN, EOS_TOKEN] + sorted(chars)
    stoi  = {ch: i for i, ch in enumerate(vocab)}
    itos  = {i: ch for i, ch in enumerate(vocab)}
    return stoi, itos


# ===========================================================================
# Data preparation  —  stream raw file → SQLite
# ===========================================================================

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id     INTEGER PRIMARY KEY,
    tokens TEXT NOT NULL                -- encoded record string, max ~300 chars
);
"""

# Number of DB rows to insert per transaction
_BATCH_INSERT = 10_000


def _download(url: str, dest: str) -> None:
    response = urllib.request.urlopen(url)
    total = int(response.headers.get("Content-Length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    chunk = 1 << 16
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
    ) as bar:
        while True:
            data = response.read(chunk)
            if not data:
                break
            f.write(data)
            bar.update(len(data))


def download_data() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {DATA_URL} ...")
        _download(DATA_URL, ZIP_PATH)
    else:
        print(f"Archive already present at {ZIP_PATH}")
    if not os.path.exists(RAW_PATH):
        print("Extracting archive ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Done.")
    else:
        print(f"Raw file already present at {RAW_PATH}")


def build_db(stoi: dict[str, int], max_rows: int | None = None) -> int:
    """
    Stream allCountries.txt directly into corpus.db.

    Each row stores the encoded record string (plain text, ≤ ~300 chars).
    No split logic here — that happens at training time from the ID list.
    Returns total number of records written.
    """
    print(f"Building corpus DB ({DB_PATH}) ...")

    con = sqlite3.connect(DB_PATH)
    con.executescript(_DB_SCHEMA)
    # WAL mode: faster bulk inserts
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")

    total_bytes = os.path.getsize(RAW_PATH)
    written = skipped = 0
    buf: list[tuple[str,]] = []

    def _flush() -> None:
        con.executemany("INSERT INTO records (tokens) VALUES (?)", buf)
        con.commit()
        buf.clear()

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
         tqdm(total=total_bytes, unit="B", unit_scale=True,
              unit_divisor=1024, desc="corpus") as bar:

        for raw_line in fin:
            bar.update(len(raw_line.encode("utf-8")))
            cols = raw_line.rstrip("\n").split("\t")
            if len(cols) < 19:
                continue

            country_code = cols[COL_COUNTRY_CODE].strip()
            feature_cls  = cols[COL_FEATURE_CLASS].strip()
            feature_code = cols[COL_FEATURE_CODE].strip()
            if not country_code:
                continue

            raw_name = cols[COL_ASCIINAME].strip() or cols[COL_NAME].strip()
            name = _normalize_name(raw_name)
            if not name:
                skipped += 1
                continue

            text = _record_text(country_code, feature_cls, feature_code, name)
            buf.append((text,))
            written += 1

            if len(buf) >= _BATCH_INSERT:
                _flush()

            if max_rows and written >= max_rows:
                break

    if buf:
        _flush()

    con.commit()
    con.close()

    print(f"DB: {written:,} records written, {skipped:,} skipped  ->  {DB_PATH}")
    return written


def prepare(max_rows: int | None = None) -> None:
    download_data()
    stoi, itos = build_vocab()
    vocab_size = len(stoi)
    single_chars = [c for c in stoi if len(c) == 1]
    emoji_chars  = [c for c in single_chars if ord(c) > 127]
    plain_chars  = [c for c in single_chars if ord(c) <= 127]
    special      = [c for c in stoi if len(c) > 1]
    print(
        f"Vocab: {vocab_size} tokens  "
        f"({len(plain_chars)} ASCII + {len(emoji_chars)} emoji + {len(special)} special)"
    )
    print(f"  Special: {special}")
    print(f"  ASCII  : {''.join(sorted(plain_chars))!r}")
    print(f"  Emoji  : {''.join(sorted(emoji_chars, key=ord))}")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    total = build_db(stoi, max_rows=max_rows)

    # persist vocab inside the DB
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS vocab (token TEXT PRIMARY KEY, id INTEGER)")
    con.executemany("INSERT OR REPLACE INTO vocab VALUES (?,?)", stoi.items())
    con.commit()
    con.close()


    # Benchmark get_batch so the user knows what to expect during training
    print("\nBenchmarking get_batch ...")
    con = sqlite3.connect(DB_PATH)
    all_ids = list(range(1, total + 1))
    # warm up
    get_batch("train", con, stoi, all_ids)
    # time 10 calls
    t0 = time.perf_counter()
    for _ in range(10):
        get_batch("train", con, stoi, all_ids)
    elapsed = (time.perf_counter() - t0) / 10
    con.close()
    print(f"get_batch average: {elapsed*1000:.1f} ms  ({1/elapsed:.0f} batches/sec)")

    print("\nData preparation complete. Run `uv run city_gpt.py train` next.")


# ===========================================================================
# DB-backed data loading
# ===========================================================================

def _load_vocab(con: sqlite3.Connection) -> tuple[dict[str, int], dict[int, str]]:
    rows = con.execute("SELECT token, id FROM vocab").fetchall()
    stoi = {tok: i for tok, i in rows}
    itos = {i: tok for tok, i in rows}
    return stoi, itos


def get_batch(
    split: str,
    con: sqlite3.Connection,
    stoi: dict[str, int],
    id_pool: list[int],      # pre-split list of DB row IDs for this split
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pick batch_size random IDs from id_pool, fetch their record strings
    with a single WHERE id IN (...) query (fast PK lookup), encode on the
    fly, concatenate into a flat token buffer, slice block_size windows.
    """
    # Fetch enough rows to fill block_size * batch_size tokens with headroom.
    # Average record is ~25 tokens; factor of 3 gives comfortable margin.
    fetch_n = batch_size * max(block_size // 25 + 1, 3)
    chosen  = random.sample(id_pool, min(fetch_n, len(id_pool)))

    placeholders = ",".join("?" * len(chosen))
    rows = con.execute(
        f"SELECT tokens FROM records WHERE id IN ({placeholders})",
        chosen,
    ).fetchall()

    flat: list[int] = []
    for (text,) in rows:
        flat.extend(stoi[ch] for ch in text if ch in stoi)

    # Guard: shouldn't happen with real data
    if len(flat) <= block_size:
        flat = flat * ((block_size + 2) // max(len(flat), 1) + 1)

    data = torch.tensor(flat, dtype=torch.long)
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i : i + block_size]         for i in ix])
    y    = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: "GPTLanguageModel",
    con: sqlite3.Connection,
    stoi: dict[str, int],
    train_ids: list[int],
    val_ids: list[int],
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, id_pool in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, con, stoi, id_pool)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ===========================================================================
# Model  — Karpathy GPT architecture, unchanged
# ===========================================================================

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: self-attention then feed-forward, both with residuals."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT (Karpathy's architecture)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx


# ===========================================================================
# CLI commands
# ===========================================================================

def cmd_prepare(args: argparse.Namespace) -> None:
    prepare(max_rows=args.max_rows)


def cmd_train(args: argparse.Namespace) -> None:
    if not os.path.exists(DB_PATH):
        print(f"Missing {DB_PATH}. Run `uv run city_gpt.py prepare` first.")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    stoi, itos = _load_vocab(con)
    vocab_size = len(stoi)

    # Load all row IDs into memory (~12M ints ≈ 96 MB), shuffle, split 90/10
    print("Loading record IDs ...")
    total = con.execute("SELECT MAX(id) FROM records").fetchone()[0]
    all_ids = list(range(1, total + 1))
    random.seed(1337)
    random.shuffle(all_ids)
    n_val     = int(total * VAL_FRACTION)
    val_ids   = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(
        f"DB: {total:,} records  "
        f"(train {len(train_ids):,} / val {len(val_ids):,})  "
        f"|  vocab size: {vocab_size}  |  device: {device}"
    )

    torch.manual_seed(1337)
    model    = GPTLanguageModel(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    bar = trange(max_iters, desc="training", unit="step")
    for step in bar:
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, con, stoi, train_ids, val_ids)
            tqdm.write(
                f"step {step:5d} / {max_iters}  |  "
                f"train loss {losses['train']:.4f}  |  "
                f"val loss {losses['val']:.4f}"
            )
            # Seed each sample from a random val record up to and including
            # the '|' separator, then let the model generate only the name.
            sos_id = stoi[SOS_TOKEN]
            eos_id = stoi[EOS_TOKEN]
            sep_id = stoi[FIELD_SEP]
            seed_rows = con.execute(
                f"SELECT tokens FROM records WHERE id IN "
                f"({','.join('?' * eval_samples)})",
                random.sample(val_ids, eval_samples),
            ).fetchall()
            samples = []
            for (text,) in seed_rows:
                # encode the full record, keep tokens up to and including '|'
                all_ids = [sos_id] + [stoi[ch] for ch in text if ch in stoi]
                sep_pos = next((i for i, t in enumerate(all_ids) if t == sep_id), None)
                prefix  = all_ids[: sep_pos + 1] if sep_pos is not None else all_ids
                ctx     = torch.tensor([prefix], dtype=torch.long, device=device)
                out     = model.generate(ctx, max_new_tokens=60)[0].tolist()
                # only show the generated part (after the prefix)
                generated = out[len(prefix):]
                if eos_id in generated:
                    generated = generated[:generated.index(eos_id)]
                prefix_str = "".join(itos.get(i, "?") for i in prefix[1:])  # skip SOS
                name_str   = "".join(itos.get(i, "?") for i in generated)
                samples.append(f"{prefix_str}{name_str}")
            tqdm.write("  samples: " + " | ".join(samples))
            torch.save(
                {"model_state": model.state_dict(), "stoi": stoi, "itos": itos},
                MODEL_PATH,
            )
            tqdm.write(f"  checkpoint saved -> {MODEL_PATH}")
        xb, yb = get_batch("train", con, stoi, train_ids)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    con.close()
    print(f"Training complete. Model saved to {MODEL_PATH}")


def cmd_generate(args: argparse.Namespace) -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}. Run train first.")
        sys.exit(1)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    encode = lambda s: [stoi[ch] for ch in s if ch in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)

    model = GPTLanguageModel(len(stoi)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sos_id = stoi[SOS_TOKEN]
    prompt = args.prompt or ""
    ctx = torch.tensor(
        [[sos_id] + encode(prompt)],
        dtype=torch.long, device=device,
    )

    out = model.generate(ctx, max_new_tokens=args.n,
                         temperature=args.temperature, top_k=args.top_k)
    print(decode(out[0].tolist()))


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="City GPT — character-level Transformer on GeoNames data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_prep = sub.add_parser("prepare", help="Download & preprocess GeoNames data")
    p_prep.add_argument(
        "--max-rows", dest="max_rows", type=int, default=None, metavar="N",
        help="Cap the corpus at N records (default: all ~12 M rows)",
    )

    sub.add_parser("train", help="Train the GPT model")

    p_gen = sub.add_parser("generate", help="Sample city records from a trained model")
    p_gen.add_argument(
        "--prompt", type=str, default="", metavar="TEXT",
        help='Seed text, e.g. "🇺🇸🏙️PPL|new" to steer generation',
    )
    p_gen.add_argument("--n",           type=int,   default=500,  metavar="N",
                       help="Number of tokens to generate (default: 500)")
    p_gen.add_argument("--temperature", type=float, default=1.0,  metavar="T",
                       help="Sampling temperature (default: 1.0)")
    p_gen.add_argument("--top-k",       type=int,   default=None, dest="top_k",
                       metavar="K",
                       help="Top-k sampling; omit to use full distribution")

    args = parser.parse_args()
    {
        "prepare":  cmd_prepare,
        "train":    cmd_train,
        "generate": cmd_generate,
    }.get(args.command, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
