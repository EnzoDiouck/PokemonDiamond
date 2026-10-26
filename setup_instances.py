"""Recopie les touches de melonDS (instance 0) vers les instances 1..N-1.

Chaque melonDS ouvert en plus est une "instance" avec ses propres touches dans melonDS.toml.
Sans ca, la 2e fenetre aurait les touches par defaut (Start = Entree au lieu de S...).

Usage (melonDS doit etre FERME) :  python setup_instances.py 4
"""
import re
import shutil
import subprocess
import sys

CONFIG = r"C:\Users\enzop\Desktop\melonDS.toml"
SECTIONS = ["Keyboard", "Joystick"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4

if "melonDS.exe" in subprocess.run(["tasklist"], capture_output=True, text=True).stdout:
    sys.exit("Ferme melonDS d'abord : il reecrit sa configuration en se fermant et effacerait les changements.")

text = open(CONFIG, encoding="utf-8").read()


def section(name):
    """Le bloc '[name]' jusqu'a la section suivante (ou None)."""
    m = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M)
    return m


shutil.copy(CONFIG, CONFIG + ".bak")
for sec in SECTIONS:
    src = section(f"Instance0.{sec}")
    if not src:
        sys.exit(f"[Instance0.{sec}] introuvable dans {CONFIG}")
    for i in range(1, n):
        block = f"[Instance{i}.{sec}]\n{src.group(1)}"
        dst = section(f"Instance{i}.{sec}")
        if dst:
            text = text[:dst.start()] + block + text[dst.end():]
        else:
            text = text.rstrip("\n") + "\n\n" + block
open(CONFIG, "w", encoding="utf-8").write(text)
print(f"Touches recopiees vers les instances 1 a {n - 1}. Sauvegarde de l'ancienne config : {CONFIG}.bak")
