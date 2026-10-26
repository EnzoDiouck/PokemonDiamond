"""Lecture de donnees dans la ROM de Pokemon Diamant : textes du jeu et noms des lieux.

Une ROM DS contient un petit systeme de fichiers (NitroFS). Les textes sont dans msgdata/msg.narc,
une archive de fichiers ; chaque fichier est une liste de phrases chiffrees.
"""
import struct

from text import decode

ROM = (r"C:\Users\enzop\Desktop\Pokemon - Diamond Version (Europe) (Rev 5)"
       r"\Pokemon - Diamond Version (Europe) (Rev 5).nds")


class NitroFS:
    def __init__(self, path=ROM):
        self.rom = open(path, "rb").read()
        fnt_off, _, fat_off, fat_size = struct.unpack_from("<4I", self.rom, 0x40)
        self.fat = [struct.unpack_from("<2I", self.rom, fat_off + 8 * i) for i in range(fat_size // 8)]
        self.paths = {}
        self._walk(fnt_off, 0xF000, "")

    def _walk(self, fnt, dir_id, prefix):
        """Table des noms : chaque dossier a une sous-table (nom, sous-dossier ou fichier)."""
        sub_off, first_file = struct.unpack_from("<IH", self.rom, fnt + 8 * (dir_id & 0xFFF))
        pos, file_id = fnt + sub_off, first_file
        while True:
            length = self.rom[pos]
            if length == 0:
                break
            name = self.rom[pos + 1:pos + 1 + (length & 0x7F)].decode("latin-1")
            pos += 1 + (length & 0x7F)
            if length & 0x80:  # sous-dossier
                sub_id = struct.unpack_from("<H", self.rom, pos)[0]
                pos += 2
                self._walk(fnt, sub_id, prefix + name + "/")
            else:
                self.paths[prefix + name] = file_id
                file_id += 1

    def file(self, path):
        start, end = self.fat[self.paths[path]]
        return self.rom[start:end]


def narc_files(data):
    """Archive NARC -> liste des fichiers qu'elle contient."""
    pos = struct.unpack_from("<H", data, 0x0C)[0]  # taille de l'en-tete
    assert data[pos:pos + 4] == b"BTAF"
    btaf_size, count = struct.unpack_from("<IH", data, pos + 4)
    entries = [struct.unpack_from("<2I", data, pos + 12 + 8 * i) for i in range(count)]
    pos += btaf_size
    pos += struct.unpack_from("<I", data, pos + 4)[0]  # saute BTNF (noms)
    assert data[pos:pos + 4] == b"GMIF"
    gmif = pos + 8
    return [data[gmif + s:gmif + e] for s, e in entries]


def msg_strings(data):
    """Fichier de textes Diamant/Perle -> liste de phrases (dechiffrees)."""
    count, seed = struct.unpack_from("<2H", data, 0)
    out = []
    for i in range(count):
        key = (seed * 0x2FD * (i + 1)) & 0xFFFF
        key = key | (key << 16)
        offset, length = struct.unpack_from("<2I", data, 4 + 8 * i)
        offset, length = offset ^ key, length ^ key
        if offset + 2 * length > len(data):
            out.append("")
            continue
        chars = list(struct.unpack_from(f"<{length}H", data, offset))
        k = (0x91BD3 * (i + 1)) & 0xFFFF
        for j in range(length):
            chars[j] ^= k
            k = (k + 0x493D) & 0xFFFF
        out.append(decode(chars + [0xFFFF]))
    return out


def all_messages(fs=None):
    fs = fs or NitroFS()
    return [msg_strings(f) for f in narc_files(fs.file("msgdata/msg.narc"))]


LOCATION_NAMES_FILE = 382  # fichier de textes des 112 noms de lieux ("Twinleaf Town", "Route 201"...)
MAP_HEADERS = 0x020EEDBC   # table des en-tetes de cartes (24 octets chacune), chargee en RAM
MAP_HEADER_SIZE = 24
OFS_LOCATION = 0x12        # dans l'en-tete : numero du nom de lieu
N_MAPS = 560


def location_names(fs=None):
    fs = fs or NitroFS()
    return msg_strings(narc_files(fs.file("msgdata/msg.narc"))[LOCATION_NAMES_FILE])


SPECIES_NAMES_FILE = 362  # noms des Pokemon, index = numero d'espece (1 = BULBASAUR)


def species_names(fs=None):
    fs = fs or NitroFS()
    return [n.capitalize() for n in msg_strings(narc_files(fs.file("msgdata/msg.narc"))[SPECIES_NAMES_FILE])]


def map_places(mem):
    """Pour chaque numero de carte, le nom du lieu (ex. 342 -> 'Route 201', 415 -> 'Twinleaf Town')."""
    names = location_names()
    table = mem.read(MAP_HEADERS, N_MAPS * MAP_HEADER_SIZE)
    return [names[struct.unpack_from("<H", table, i * MAP_HEADER_SIZE + OFS_LOCATION)[0]] for i in range(N_MAPS)]


if __name__ == "__main__":
    fs = NitroFS()
    print(len(fs.paths), "fichiers dans la ROM ; msg.narc :", "msgdata/msg.narc" in fs.paths)
    for i, strings in enumerate(all_messages(fs)):
        if "Twinleaf Town" in strings and "Route 201" in strings:
            print(f"Fichier de textes {i} : {len(strings)} lieux, ex. {strings[:8]}")
