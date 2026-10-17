"""Token vocabularies shared by every model.

Ids 0-2 are reserved. Evaluation always renormalises predictions over the
symbol alphabet only (20 amino acids or 4 bases), so every model -- n-gram,
LSTM, Transformer -- is scored on exactly the same event space.
"""

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3

AMINO = "ACDEFGHIKLMNPQRSTVWY"
BASES = "ACGT"

ALPHABETS = {"protein": AMINO, "dna": BASES}


def vocab_size(kind: str) -> int:
    return N_SPECIAL + len(ALPHABETS[kind])


def symbol_ids(kind: str) -> list[int]:
    return list(range(N_SPECIAL, N_SPECIAL + len(ALPHABETS[kind])))


def encode_table(kind: str):
    import numpy as np

    table = np.full(256, 255, dtype=np.uint8)  # 255 = invalid
    for i, ch in enumerate(ALPHABETS[kind]):
        table[ord(ch)] = N_SPECIAL + i
        table[ord(ch.lower())] = N_SPECIAL + i
    return table


def decode(kind: str, ids) -> str:
    alpha = ALPHABETS[kind]
    return "".join(alpha[i - N_SPECIAL] for i in ids if i >= N_SPECIAL)
