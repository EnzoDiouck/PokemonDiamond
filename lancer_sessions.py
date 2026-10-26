"""Ouvre N fenetres melonDS avec Pokemon Diamant et charge la sauvegarde rapide 1 dans chacune.

Usage :  python lancer_sessions.py 4      puis      python train.py --voir
(Une seule fois avant, melonDS ferme : python setup_instances.py 4)
"""
import sys
import time

from melonds import MelonDS, find_windows, launch_instance

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for _ in range(n - len(find_windows())):
    launch_instance()

time.sleep(1)
windows = find_windows()
for hwnd, _ in windows:
    MelonDS(hwnd).load_state(1)
print(f"{len(windows)} fenetre(s) melonDS prete(s). Lance maintenant : python train.py --voir")
