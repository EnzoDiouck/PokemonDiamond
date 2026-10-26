"""Tableau de bord en direct : UNE fenetre decoupee en cases, une case par fenetre DS
(ce que l'IA voit, la touche qu'elle appuie, ses hesitations).
Option : l'ecran du jeu en couleur d'UNE fenetre choisie. Option : video .mp4 de cette fenetre.
"""
import math

import cv2
import numpy as np

from env import ACTIONS

CELL_W, CELL_H = 600, 400  # une case = une fenetre DS
GAME_W, GAME_H = 448, 336  # chaque ecran du jeu (option --capture) : haut + bas
GAME_PANEL_H = 30 + 2 * GAME_H + 34  # hauteur de la colonne des 2 ecrans
SIDE_W = GAME_W + 20       # colonne de droite qui l'accueille
BG = (32, 28, 26)          # couleurs en BGR (OpenCV)
PANEL = (52, 46, 42)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
LIT = (80, 200, 255)       # touche appuyee
ACCENT = (120, 220, 120)
RED = (60, 60, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE = "IA Pokemon Diamant"


def _text(img, s, x, y, scale=0.45, color=TEXT, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _gray_box(img, gray, x, y, w, h, label):
    img[y:y + h, x:x + w] = cv2.cvtColor(cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST),
                                         cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    _text(img, label, x + 4, y + 14, 0.42, LIT)


def _pad(img, pressed, ox, oy):
    """Mini manette : croix + X/A/B + START, la touche choisie s'allume."""
    s = 22
    cx, cy = ox + 35, oy + 45
    cv2.rectangle(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), PANEL, -1)
    arrows = {"UP": ((cx, cy - s), "^"), "DOWN": ((cx, cy + s), "v"),
              "LEFT": ((cx - s, cy), "<"), "RIGHT": ((cx + s, cy), ">")}
    for name, ((x, y), sym) in arrows.items():
        on = name == pressed
        cv2.rectangle(img, (x - s // 2, y - s // 2), (x + s // 2, y + s // 2), LIT if on else PANEL, -1)
        _text(img, sym, x - 5, y + 5, 0.5, BG if on else DIM, 2)
    for name, (x, y) in {"X": (ox + 115, oy + 18), "A": (ox + 145, oy + 45), "B": (ox + 115, oy + 72)}.items():
        on = name == pressed
        cv2.circle(img, (x, y), 14, LIT if on else PANEL, -1)
        _text(img, name, x - 6, y + 6, 0.5, BG if on else DIM, 2)
    on = pressed == "START"
    cv2.rectangle(img, (ox + 95, oy + 92), (ox + 165, oy + 110), LIT if on else PANEL, -1)
    _text(img, "START", ox + 106, oy + 106, 0.42, BG if on else DIM)


def _bars(img, probs, action, ox, oy):
    """Probabilite de chaque touche : la longueur montre ce que le cerveau a envie de faire."""
    _text(img, "Probabilites", ox, oy, 0.45, DIM)
    bw = 80
    for i, (name, p) in enumerate(zip(ACTIONS, probs)):
        y = oy + 10 + i * 24
        on = i == action
        _text(img, name, ox, y + 14, 0.42, LIT if on else TEXT)
        cv2.rectangle(img, (ox + 52, y), (ox + 52 + bw, y + 17), PANEL, -1)
        cv2.rectangle(img, (ox + 52, y), (ox + 52 + int(bw * p), y + 17), LIT if on else ACCENT, -1)
        _text(img, f"{p * 100:3.0f}%", ox + 137, y + 14, 0.42)


def draw_cell(index, env, probs, action, value, ep_reward, info, highlight=False):
    """La case d'une fenetre DS (600 x 400)."""
    img = np.full((CELL_H, CELL_W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (CELL_W - 1, CELL_H - 1), LIT if highlight else PANEL, 1)
    _text(img, f"Fenetre {index + 1}  |  Action {env.steps}/{env.max_steps}", 10, 20, 0.5, TEXT)

    # Colonne 1 : ce que l'IA voit, et sa memoire des cases visitees (zoom sur le centre)
    _gray_box(img, env.frames[-1], 10, 30, 192, 144, "Vision de l'IA")
    _gray_box(img, env.last_map[18:54, 24:72], 10, 184, 192, 144, "Memoire (cases vues)")
    cv2.rectangle(img, (104, 254), (109, 259), LIT, -1)  # elle, au centre

    # Colonne 2 : la touche appuyee + les chiffres
    _pad(img, ACTIONS[action], 215, 30)
    lines = [
        (f"Cases : {info['tiles']}", ACCENT),
        (f"Cartes : {info['maps']} (n.{info['pos'][0]})", ACCENT),
        (f"Equipe : {info['party']} (niv. {info['levels']})", LIT),
        (f"Badges : {info['badges']}", LIT),
        (f"Points : {ep_reward:.1f}", ACCENT),
        (f"Prevus : {value:.1f}", DIM),
    ]
    if env.read_text:
        lines.append((f"Dialogues : {info['texts']} (relus {info['repeats']})", DIM))
    if info.get("options"):
        lines.append(("Dans OPTIONS !", RED))
    for i, (s, c) in enumerate(lines):
        _text(img, s, 215, 175 + i * 21, 0.45, c)

    # Colonne 3 : les probabilites du cerveau, puis l'equipe
    _bars(img, probs, action, 405, 40)
    _team(img, info.get("team") or [], 405, 262)

    # En bas : le coach et le dernier dialogue lu
    cv2.line(img, (10, 345), (CELL_W - 10, 345), PANEL, 1)
    coach = env.coach
    if coach:  # deux lignes : le plan, puis ses lieux cibles et comment le jeu saura qu'il est fini
        if coach.objective:
            targets = ", ".join(("[ATTEINT] " if t in env.coach_rewarded else "") + t for t in coach.targets)
            rows = [(f"Plan : {coach.objective}", LIT),
                    (f"Lieux : {targets or '-'} | fini quand : {coach.done_text()}", LIT)]
        else:
            rows = [("Plan : aucun" + (" (coach desactive)" if coach.disabled else ""), DIM),
                    (f"En attente : {len(coach.pending)} scene(s)", DIM)]
        for row, (line, color) in enumerate(rows):
            _text(img, line[:95], 10, 365 + row * 21, 0.45, color)
    else:
        _text(img, "Coach desactive (--sans-coach)", 10, 365, 0.45, DIM)
        if env.last_lines:
            _text(img, f"Dialogue : {env.last_lines[-1]}"[:100], 10, 388, 0.42, DIM)
    return img


_SPECIES = None


def _species_name(n):
    """Nom du Pokemon (lu une fois dans la ROM)."""
    global _SPECIES
    if _SPECIES is None:
        try:
            from rom_data import species_names
            _SPECIES = species_names()
        except Exception:
            _SPECIES = []
    return _SPECIES[n] if 0 < n < len(_SPECIES) else f"#{n}"


def _team(img, team, ox, oy):
    """L'equipe : nom, niveau et barre de PV de chaque Pokemon."""
    _text(img, "Equipe", ox, oy, 0.45, DIM)
    if not team:
        _text(img, "aucun Pokemon", ox, oy + 18, 0.4, DIM)
        return
    for i, (species, level, hp, max_hp) in enumerate(team):
        y = oy + 8 + i * 12
        _text(img, f"{_species_name(species)[:10]} N.{level}", ox, y + 9, 0.36, TEXT)
        frac = hp / max_hp if max_hp else 0.0
        color = ACCENT if frac > 0.5 else LIT if frac > 0.2 else RED
        cv2.rectangle(img, (ox + 110, y + 2), (ox + 180, y + 9), PANEL, -1)
        cv2.rectangle(img, (ox + 110, y + 2), (ox + 110 + int(70 * frac), y + 9), color, -1)


def draw_game(index, env, height):
    """La colonne de droite : les 2 ecrans du jeu en couleur de la fenetre choisie."""
    img = np.full((height, SIDE_W, 3), BG, np.uint8)
    _text(img, f"Jeu - fenetre {index + 1}", 10, 20, 0.5, LIT)
    img[30:30 + GAME_H, 10:10 + GAME_W] = cv2.resize(env.last_color[:, :, ::-1], (GAME_W, GAME_H),
                                                     interpolation=cv2.INTER_NEAREST)
    bottom = getattr(env, "last_bottom", None)
    if bottom is not None and bottom.size:
        y = 30 + GAME_H + 24
        _text(img, "Ecran du bas", 10, y - 6, 0.42, DIM)
        img[y:y + GAME_H, 10:10 + GAME_W] = cv2.resize(np.ascontiguousarray(bottom[:, :, ::-1]), (GAME_W, GAME_H),
                                                       interpolation=cv2.INTER_NEAREST)
    return img


class Dashboard:
    def __init__(self, n_windows, focus=0, capture=False, show=True, record_path=None, fps=10):
        """focus = la fenetre DS mise en avant (capture du jeu et video)."""
        self.n, self.focus, self.capture, self.show = n_windows, focus, capture, show
        self.cols = 1 if n_windows == 1 else 2
        self.rows = math.ceil(n_windows / self.cols)
        # (avec --capture, assez haut pour les 2 ecrans du jeu)
        self.height = max(self.rows * CELL_H, GAME_PANEL_H if capture else 0)
        if show:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)  # redimensionnable a la souris
            cv2.resizeWindow(TITLE, self.cols * CELL_W + (SIDE_W if capture else 0), self.height)
        self.writer = None
        if record_path:  # la video : la case de la fenetre choisie (+ son jeu si --capture)
            size = (CELL_W + (SIDE_W if capture else 0), max(CELL_H, GAME_PANEL_H) if capture else CELL_H)
            self.writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def update(self, envs, probs, actions, values, ep_rewards, infos, episode, total_steps):
        cells = [draw_cell(i, envs[i], probs[i], actions[i], values[i], ep_rewards[i], infos[i],
                           highlight=(i == self.focus and self.n > 1))
                 for i in range(self.n)]
        game = draw_game(self.focus, envs[self.focus], max(CELL_H, GAME_PANEL_H)) if self.capture else None

        if self.show:
            grid = np.full((self.height, self.cols * CELL_W, 3), BG, np.uint8)
            for i, cell in enumerate(cells):
                r, c = divmod(i, self.cols)
                grid[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell
            if game is not None:
                side = np.full((grid.shape[0], SIDE_W, 3), BG, np.uint8)
                side[:min(game.shape[0], grid.shape[0])] = game[:grid.shape[0]]
                grid = np.hstack([grid, side])
            cv2.imshow(TITLE, grid)
            cv2.setWindowTitle(TITLE, f"{TITLE}  -  episode {episode + 1}  -  {total_steps} actions")
            cv2.waitKey(1)

        if self.writer:
            frame = cells[self.focus]
            if game is not None:
                cell = np.full((game.shape[0], CELL_W, 3), BG, np.uint8)
                cell[:CELL_H] = frame
                frame = np.hstack([cell, game])
            self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
        if self.show:
            cv2.destroyWindow(TITLE)
