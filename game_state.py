"""Ce que l'IA sait du jeu, lu dans la RAM de Pokemon Diamant (version anglaise, ADAE)."""
import struct

import numpy as np

from memory import DSMemory

SAVE_PTR = 0x02106FAC   # pointeur vers les donnees du jeu en RAM
SAVE_BLOCK = 0xD214     # debut du bloc de sauvegarde "general" (relatif au pointeur)

# Offsets dans le bloc de sauvegarde (structure de sauvegarde Diamant/Perle)
OFS_NAME = 0x64         # nom du dresseur (8 caracteres, alphabet du jeu)
OFS_TID = 0x74          # numero de dresseur
OFS_MONEY = 0x78
OFS_BADGES = 0x7E       # 1 bit par badge
OFS_PARTY_COUNT = 0x94
OFS_PARTY = 0x98        # 6 Pokemon x 236 octets
PKM_SIZE = 236
OFS_MAP = 0x1238        # numero de la carte (415 = chambre du heros)
OFS_X = 0x1240          # position en cases (locale en interieur, globale dehors)
OFS_Y = 0x1244
# Flags d'evenements (1 bit = "Matt est parti", "starter recu", "dresseur battu"...).
# Offsets de PKHeX (SAV4DP : EventWork 0xD9C = 288 variables de 16 bits, puis EventFlag 0xFDC).
# Trainer1 0x64 / Party 0x98 de la meme table correspondent a ce qu'on a verifie : bon signe, mais
# a confirmer en jeu avec "python flags.py avant" / "python flags.py apres" (voir flags.py).
OFS_FLAGS = 0xFDC
NUM_FLAGS = 2912        # 0x16C octets

# Les 4 blocs d'un Pokemon (A: espece/objet/dresseur, B: attaques, C: surnom, D: divers)
# sont ranges dans l'un de ces 24 ordres, choisi d'apres le PID du Pokemon.
BLOCK_ORDERS = ["ABCD", "ABDC", "ACBD", "ACDB", "ADBC", "ADCB", "BACD", "BADC", "BCAD", "BCDA",
                "BDAC", "BDCA", "CABD", "CADB", "CBAD", "CBDA", "CDAB", "CDBA", "DABC", "DACB",
                "DBAC", "DBCA", "DCAB", "DCBA"]


def _decrypt(words, seed):
    """Le jeu chiffre avec un generateur pseudo-aleatoire : refaire la meme suite et XOR = dechiffrer."""
    out = []
    for w in words:
        seed = (seed * 0x41C64E6D + 0x6073) & 0xFFFFFFFF
        out.append(w ^ (seed >> 16))
    return out


def decode_pokemon(raw, trainer_id=None):
    """236 octets chiffres -> dict (espece, niveau, PV...), ou None si vide / illisible."""
    pid, _, checksum = struct.unpack_from("<IHH", raw, 0)
    if pid == 0 and checksum == 0:
        return None
    data = _decrypt(struct.unpack_from("<64H", raw, 0x08), checksum)
    if sum(data) & 0xFFFF != checksum:
        return None  # donnees en cours d'ecriture par le jeu, ou mauvais dechiffrement

    order = BLOCK_ORDERS[((pid & 0x3E000) >> 13) % 24]
    chunk = lambda i: data[16 * i:16 * i + 16]
    # Lecture 1 : la i-eme position contient le bloc order[i]
    blocks = {order[i]: chunk(i) for i in range(4)}
    if trainer_id is not None and blocks["A"][2] != trainer_id:
        # Les sources different sur le sens de la table : lecture 2 = permutation inverse.
        # On garde celle ou le numero du dresseur d'origine est le notre.
        inverse = {"ABCD"[i]: chunk("ABCD".index(order[i])) for i in range(4)}
        if inverse["A"][2] == trainer_id:
            blocks = inverse

    # Partie "combat" (niveau, PV), chiffree avec le PID
    battle = _decrypt(struct.unpack_from("<50H", raw, 0x88), pid)
    return {
        "species": blocks["A"][0],
        "ot_id": blocks["A"][2],
        "level": battle[2] & 0xFF,
        "hp": battle[3],
        "max_hp": battle[4],
    }


class GameState:
    def __init__(self, mem: DSMemory):
        self.mem = mem

    def _save(self):
        return self.mem.u32(SAVE_PTR) + SAVE_BLOCK

    def position(self):
        """(carte, x, y)"""
        s = self._save()
        return self.mem.u16(s + OFS_MAP), self.mem.u32(s + OFS_X), self.mem.u32(s + OFS_Y)

    def trainer_name(self):
        """Alphabet du jeu : 0x12B = 'A', 0x145 = 'a', 0x121 = '0', 0xFFFF = fin."""
        chars = []
        for c in struct.unpack("<8H", self.mem.read(self._save() + OFS_NAME, 16)):
            if c == 0xFFFF:
                break
            if 0x12B <= c < 0x12B + 26:
                chars.append(chr(ord("A") + c - 0x12B))
            elif 0x145 <= c < 0x145 + 26:
                chars.append(chr(ord("a") + c - 0x145))
            elif 0x121 <= c < 0x121 + 10:
                chars.append(chr(ord("0") + c - 0x121))
            else:
                chars.append("?")
        return "".join(chars)

    def event_flags(self, ram=None):
        """Tableau de 0/1 (un par flag d'evenement). ram = copie de toute la RAM (snapshot) si on l'a deja :
        evite une lecture de plus. None si le pointeur est illisible (pendant un chargement)."""
        if ram is not None:
            return event_flags(ram)
        raw = self.mem.read(self._save() + OFS_FLAGS, NUM_FLAGS // 8)  # lecture directe : 364 octets
        if not raw or len(raw) != NUM_FLAGS // 8:
            return None
        return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")

    def money(self):
        return self.mem.u32(self._save() + OFS_MONEY)

    def badges(self):
        """Nombre de badges (0 a 8)."""
        return bin(self.mem.u8(self._save() + OFS_BADGES)).count("1")

    def party(self):
        """Liste des Pokemon de l'equipe (dicts), dans l'ordre."""
        s = self._save()
        count = min(self.mem.u8(s + OFS_PARTY_COUNT), 6)
        if count == 0:
            return []
        raw = self.mem.read(s + OFS_PARTY, PKM_SIZE * count)
        tid = self.mem.u16(s + OFS_TID)
        team = [decode_pokemon(raw[i * PKM_SIZE:(i + 1) * PKM_SIZE], tid) for i in range(count)]
        return [p for p in team if p]

    def summary(self):
        """Ce dont les recompenses ont besoin, en une seule lecture."""
        team = self.party()
        return {
            "party": len(team),
            "levels": sum(p["level"] for p in team),
            "hp": sum(p["hp"] for p in team),
            "max_hp": sum(p["max_hp"] for p in team),
            "team": [(p["species"], p["level"], p["hp"], p["max_hp"]) for p in team],  # pour l'affichage
            "badges": self.badges(),
        }


def event_flags(ram):
    """Flags d'evenements lus dans une copie de la RAM (numpy uint8, 4 Mo). Flag i = octet i // 8, bit i % 8."""
    ptr = int(ram[0x106FAC:0x106FB0].view(np.uint32)[0]) & 0x3FFFFF
    start = ptr + SAVE_BLOCK + OFS_FLAGS
    if ptr == 0 or start + NUM_FLAGS // 8 > len(ram):
        return None
    return np.unpackbits(ram[start:start + NUM_FLAGS // 8], bitorder="little")


if __name__ == "__main__":
    from melonds import find_windows

    for hwnd, _ in find_windows():
        state = GameState(DSMemory(hwnd))
        print(f"{state.trainer_name()} | carte, x, y = {state.position()} | {state.money()} P "
              f"| badges {state.badges()} | equipe {state.party()}")
