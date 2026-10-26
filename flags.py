"""Verifier les flags d'evenements en jeu (fenetre melonDS n.1).

  python flags.py          -> nombre de flags allumes, et leur liste
  python flags.py avant    -> photo de la RAM (avant une scene, ex. avant de parler a Matt)
  python flags.py apres    -> ce qui a change depuis la photo :
                              - les flags passes a 1 / a 0 (zone EventFlag de game_state.py)
                              - les octets changes ailleurs dans le bloc de sauvegarde
Si la scene allume des bits DANS la zone des flags : l'offset est bon. Si les seuls changements
d'octets "un bit a la fois" sont ailleurs, l'offset OFS_FLAGS de game_state.py est a corriger.
Les flags deja vus repasser a 0 pendant l'entrainement (volatils) sont dans checkpoints/visites.json.
"""
import json
import os
import sys

import numpy as np

from game_state import NUM_FLAGS, OFS_FLAGS, SAVE_BLOCK, event_flags
from melonds import find_windows
from memory import DSMemory

PHOTO = "checkpoints/flags_avant.npy"
BLOCK_SIZE = 0xC100  # taille du bloc de sauvegarde "general" de Diamant/Perle


def save_block(ram):
    ptr = int(ram[0x106FAC:0x106FB0].view(np.uint32)[0]) & 0x3FFFFF
    return ram[ptr + SAVE_BLOCK: ptr + SAVE_BLOCK + BLOCK_SIZE].copy()


def volatile():
    path = "checkpoints/visites.json"
    return set(json.load(open(path)).get("volatile_flags", [])) if os.path.exists(path) else set()


windows = find_windows()
if not windows:
    raise SystemExit("Aucune fenetre melonDS trouvee.")
ram = DSMemory(windows[0][0]).snapshot()
bits = event_flags(ram)
mode = sys.argv[1] if len(sys.argv) > 1 else ""

if mode == "avant":
    os.makedirs("checkpoints", exist_ok=True)
    np.save(PHOTO, save_block(ram))
    print(f"Photo prise ({int(bits.sum())} flags allumes). Joue la scene, puis : python flags.py apres")
elif mode == "apres":
    old, new = np.load(PHOTO), save_block(ram)
    old_bits = np.unpackbits(old[OFS_FLAGS:OFS_FLAGS + NUM_FLAGS // 8], bitorder="little")
    vol = volatile()
    on = np.nonzero(bits & ~old_bits)[0].tolist()
    off = np.nonzero(old_bits & ~bits)[0].tolist()
    print(f"Flags passes a 1 : {on}" + (f"  (dont volatils : {sorted(set(on) & vol)})" if set(on) & vol else ""))
    print(f"Flags passes a 0 : {off}")
    changed = np.nonzero(old != new)[0]
    outside = [o for o in changed if not OFS_FLAGS <= o < OFS_FLAGS + NUM_FLAGS // 8]
    print(f"\nOctets changes dans le bloc : {len(changed)}, dont {len(outside)} hors de la zone des flags :")
    for o in outside[:40]:
        diff = int(old[o]) ^ int(new[o])
        one_bit = " <- un seul bit passe a 1" if bin(diff).count("1") == 1 and new[o] & diff else ""
        print(f"  +0x{o:04X} : {old[o]:3d} -> {new[o]:3d}{one_bit}")
else:
    vol = volatile()
    lit = np.nonzero(bits)[0].tolist()
    print(f"{len(lit)} flags allumes ({len(set(lit) & vol)} volatils) :")
    print(lit)
