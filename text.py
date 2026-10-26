"""Alphabet des Pokemon 4e generation : chaque caractere est un nombre de 16 bits."""
import numpy as np

CHARS = {}
CHARS.update({0x121 + i: str(i) for i in range(10)})
CHARS.update({0x12B + i: chr(ord("A") + i) for i in range(26)})
CHARS.update({0x145 + i: chr(ord("a") + i) for i in range(26)})
CHARS.update({
    0x1AB: "!", 0x1AC: "?", 0x1AD: ",", 0x1AE: ".", 0x1AF: "...", 0x1B0: "·", 0x1B1: "/",
    0x1B2: "'", 0x1B3: "'", 0x1B4: '"', 0x1B5: '"', 0x1B6: '"', 0x1B7: "<<", 0x1B8: ">>",
    0x1B9: "(", 0x1BA: ")", 0x1BB: "(M)", 0x1BC: "(F)", 0x1BD: "+", 0x1BE: "-", 0x1BF: "*",
    0x1C0: "#", 0x1C1: "=", 0x1C2: "&", 0x1C3: "~", 0x1C4: ":", 0x1C5: ";",
    0x1DE: " ",
    0x188: "e",  # é de "Pokémon" (en ASCII pour l'affichage OpenCV)
    0x1A8: "P",  # symbole Pokédollar (sinon "fining you 1 million" coupait la replique de Matt en deux)
    0x1E2: " ",  # espace special (poids du Pokédex : "15.2 lbs")
})
NEWLINE, SCROLL, PARAGRAPH, VAR, END = 0xE000, 0x25BC, 0x25BD, 0xFFFE, 0xFFFF


_CODES = {ch: code for code, ch in CHARS.items() if len(ch) == 1}


def encode(text):
    """Texte -> octets tels qu'ils sont en RAM (pour chercher une phrase precise)."""
    return np.array([_CODES[ch] for ch in text], np.uint16).tobytes()


def decode(words):
    """Liste de caracteres du jeu -> texte. S'arrete au premier 0xFFFF."""
    out = []
    i = 0
    while i < len(words):
        c = words[i]
        if c == END:
            break
        if c in (NEWLINE, SCROLL, PARAGRAPH):
            out.append(" ")
        elif c == VAR:  # variable non remplacee (nom, nombre...) : 0xFFFE, code, nb parametres, ...
            i += 2 + (words[i + 2] if i + 2 < len(words) else 0)
        else:
            out.append(CHARS.get(c, "?"))
        i += 1
    return " ".join("".join(out).split())


# Table de correspondance : _IS_TEXT[code] = True si le code est un caractere (bien plus rapide que np.isin)
_IS_TEXT = np.zeros(65536, bool)
_IS_TEXT[list(set(CHARS) | {NEWLINE, SCROLL, PARAGRAPH})] = True


def find_strings(ram, min_chars=12):
    """Toutes les phrases du jeu presentes dans la RAM : suites d'au moins min_chars caracteres valides
    terminees par 0xFFFF. ram = octets de la RAM (numpy uint8). Renvoie {adresse DS: texte}."""
    w = ram[: len(ram) // 2 * 2].view(np.uint16)
    # Candidats : un 0xFFFF precede d'au moins min_chars caracteres valides
    ends = np.nonzero(w == END)[0]
    ends = ends[ends >= min_chars]
    for k in range(1, min_chars + 1):
        ends = ends[_IS_TEXT[w[ends - k]]]
    out = {}
    for e in ends:
        s = e - min_chars
        while s > 0 and _IS_TEXT[w[s - 1]]:  # remonte jusqu'au debut de la phrase
            s -= 1
        txt = decode(list(w[s:e]) + [END])
        if sum(ch.isalpha() for ch in txt) >= min_chars // 2:  # evite les donnees qui ressemblent a du texte
            out[0x02000000 + 2 * int(s)] = txt
    return out


class TextTracker:
    """Retient les phrases deja vues ; new_lines() renvoie seulement les vraiment nouvelles.
    Le jeu garde parfois des morceaux d'une meme phrase a plusieurs endroits : un morceau
    d'une phrase deja vue (ou l'inverse) ne compte pas comme nouveau."""

    def __init__(self):
        self.seen = set()

    def reset(self, ram):
        """Tout le texte deja en memoire au depart (tele, restes...) est considere comme vu."""
        self.seen = set(find_strings(ram).values())

    def new_lines(self, ram, strings=None):
        """strings = find_strings(ram) deja calcule (evite de fouiller la RAM deux fois)."""
        fresh = []
        for txt in (strings if strings is not None else find_strings(ram)).values():
            if txt in self.seen:
                continue
            if not any(ch.islower() for ch in txt):  # libelles de menu ("BATTLE ITEMS") : pas un dialogue
                self.seen.add(txt)
                continue
            if any(txt in old or old in txt for old in self.seen):
                self.seen.add(txt)
                continue
            self.seen.add(txt)
            fresh.append(txt)
        return fresh
