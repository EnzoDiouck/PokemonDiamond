"""Ecriture des logs CSV.

append_csv() accepte de nouvelles colonnes : l'en-tete est complete et les anciennes lignes
gardent une case vide, comme ca on peut enrichir les logs sans perdre l'historique.
"""
import csv
import os
import threading
import time

_lock = threading.Lock()


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_csv(path, row):
    with _lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = []
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
        new_cols = [k for k in row if k not in header]
        if new_cols and header:  # nouvelles colonnes : on reecrit le fichier avec l'en-tete complet
            with open(path, newline="", encoding="utf-8") as f:
                old = list(csv.DictReader(f))
            header += new_cols
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(old)
        elif not header:
            header = list(row)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writeheader()
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header, restval="").writerow(row)
