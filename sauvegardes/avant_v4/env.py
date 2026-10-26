"""L'environnement : le pont entre Pokemon Diamant (dans melonDS) et l'IA.

reset()      -> recharge la sauvegarde rapide de depart, renvoie la premiere observation
step(action) -> appuie sur une touche, renvoie (observation, recompense, fini, infos)
"""
import json
import math
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np

from game_state import GameState
from melonds import MelonDS
from memory import DSMemory
from coach import speaker_of
from text import TextTracker, encode

# Les 6 premieres sont celles du cerveau v2 : on n'ajoute qu'a la fin (voir PPO.load)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "X", "START"]  # X = menu dans Diamant
MOVES = {"UP", "DOWN", "LEFT", "RIGHT"}

OBS_W, OBS_H = 96, 72   # image vue par l'IA (l'ecran DS fait 256x192)
FRAME_STACK = 3         # nombre d'images empilees pour percevoir le mouvement
MAP_W, MAP_H = 48, 36   # carte des cases visitees : 48 x 36 cases autour du joueur (2x2 pixels par case)
OBS_CHANNELS = FRAME_STACK + 1  # 3 images du jeu + 1 carte des cases visitees

# Attentes comptees en IMAGES du jeu (pas en secondes) : plus le jeu va vite (avance rapide, JIT),
# plus elles sont courtes en temps reel, et le jeu se comporte pareil.
STEP_TIMEOUT_FRAMES = 15  # une fleche maintenue sans que la position change : c'est un mur
STEP_SETTLE_FRAMES = 4    # fin de l'animation du pas
BUTTON_HOLD_FRAMES = 3    # duree d'appui sur A / B / X / START
BUTTON_SETTLE_FRAMES = 7  # laisse le texte / menu reagir
FPS_REFRESH_STEPS = 32    # on relit la vitesse du jeu (titre de la fenetre) toutes les 32 actions

# Echelle : 1 case = 0.1 point, le reste se compare a ca (dialogue = 10 cases, carte = 20 cases...)
# Curiosite qui dure d'un episode a l'autre : une case (ou une carte) rapporte selon le nombre
# d'episodes, toutes fenetres confondues, qui y sont DEJA passes : x 1/racine(1 + n).
# Jamais vue : 0.1 ; vue 100 fois : 0.01. Les zones rares (sorties de la ville...) restent payantes.
REWARD_NEW_TILE = 0.1   # premiere visite d'une case dans l'episode, x la nouveaute
REWARD_NEW_PLACE = 5.0  # nouveau LIEU (nouveau nom : "Route 201", "Sandgem Town"...) : sortir de la ville paie
REWARD_NEW_MAP = 1.0    # nouvelle carte dans un lieu deja vu (une maison, un etage...), x la nouveaute


class VisitCounts:
    """Combien d'episodes (toutes fenetres) sont passes par chaque case / carte, depuis le debut.
    Partage par les fenetres et sauvegarde sur disque : la curiosite survit aux redemarrages."""

    def __init__(self, path=None):
        self.path = path
        self.tiles = defaultdict(dict)  # carte -> {(x, y): nombre d'episodes}
        self.maps = {}                  # carte -> nombre d'episodes
        self.lines = {}                 # replique de dialogue -> nombre d'episodes ou elle a ete lue
        if path and os.path.exists(path):
            saved = json.load(open(path))
            for key, n in saved.get("tiles", {}).items():
                m, x, y = map(int, key.split(","))
                self.tiles[m][(x, y)] = n
            self.maps = {int(k): n for k, n in saved.get("maps", {}).items()}
            self.lines = dict(saved.get("lines", {}))

    def novelty_tile(self, pos):
        """Renvoie la nouveaute de la case (1 si jamais vue) puis compte ce passage."""
        m, x, y = pos
        n = self.tiles[m].get((x, y), 0)
        self.tiles[m][(x, y)] = n + 1
        return 1.0 / math.sqrt(1 + n)

    def novelty_map(self, map_id):
        n = self.maps.get(map_id, 0)
        self.maps[map_id] = n + 1
        return 1.0 / math.sqrt(1 + n)

    def novelty_line(self, line):
        """Meme principe pour les dialogues : la Wii ou la tele lues a chaque episode ne rapportent plus."""
        n = self.lines.get(line, 0)
        self.lines[line] = n + 1
        return 1.0 / math.sqrt(1 + n)

    def save(self):
        if not self.path:
            return
        tiles = {f"{m},{x},{y}": n for m, d in list(self.tiles.items()) for (x, y), n in list(d.items())}
        tmp = self.path + ".tmp"
        json.dump({"tiles": tiles, "maps": {str(k): v for k, v in self.maps.items()},
                   "lines": dict(self.lines)}, open(tmp, "w"))
        os.replace(tmp, self.path)  # jamais de fichier a moitie ecrit
REWARD_NEW_POKEMON = 5.0  # un Pokemon de plus dans l'equipe (le starter, une capture)
REWARD_LEVEL = 1.0        # par niveau gagne (somme des niveaux de l'equipe)
REWARD_BADGE = 20.0       # par badge
PENALTY_BLACKOUT = -3.0   # toute l'equipe K.O.
# Priorite : se deplacer d'abord (~200 cases = ~20 pts/episode), parler juste en dessous (max 20 x 0.5 = 10 pts)
REWARD_NEW_TEXT = 0.5     # phrase de dialogue jamais vue dans l'episode (= 5 cases)
MAX_TEXT_REWARDS = 20     # plafond par episode, pour ne pas lire les descriptions d'objets en boucle
REWARD_STORY_TEXT = 1.0   # en plus, si le coach a classe la phrase "histoire" (quete A ou etape S)
REWARD_QUEST_TALK = 2.0   # en plus, si c'est un personnage nomme dans la quete / l'etape qui parle
REWARD_QUEST = 3.0          # coach : quete principale (lieu atteint ou phrase-preuve relue, 1 fois/episode)
REWARD_STEP = 1.0           # coach : etape (la prochaine chose a faire), meme principe
PENALTY_SAVE = -2.0         # sauvegarder dans le jeu : inutile (on a les sauvegardes rapides) et ecrase ta partie
SAVE_TEXTS = ("save the game", "saved the game", "overwrite")  # phrases du menu SAVE
PENALTY_BUMP = -0.02        # une fleche qui ne fait pas bouger (mur, arbre...) : action perdue
PENALTY_STUCK = -3.0        # episode arrete pour blocage : rester sans rien faire coute
PENALTY_EARLY_MENU = -0.05  # X ou START sans aucun Pokemon : le menu ne sert a rien (sinon elle y 'attend')
STUCK_STEPS = 150           # 150 actions sans rien de nouveau (case, carte, dialogue) : l'episode s'arrete
PENALTY_OPTIONS = -10.0     # entrer dans le menu OPTIONS : rien a y faire (et y changer les boutons casserait tout)...
OPTIONS_ENDS_EPISODE = True # ...et l'episode s'arrete aussitot : la fenetre ne reste jamais coincee dedans
PENALTY_OPTIONS_STEP = -0.1  # chaque action passee dedans
OPTIONS_SIGNATURE = encode("START = X")  # texte en RAM a l'ouverture d'OPTIONS (pas toujours ensuite)...
# ...donc on compare aussi l'image a une reference de l'ecran OPTIONS (ecart ~0 dedans, ~170 ailleurs)
_ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references", "options.png")
OPTIONS_REF = cv2.imread(_ref_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(_ref_path) else None
OPTIONS_MAX_DIFF = 25
PENALTY_REPEAT_DIALOGUE = -0.25  # une boite de dialogue s'ouvre mais rien de nouveau : dialogue deja lu...
MAX_REPEAT_FACTOR = 5            # ...et c'est de pire en pire : -0.25, -0.5... (max -1.25) au meme endroit
DIALOGUE_GRACE = 3              # actions laissees au texte pour apparaitre en memoire


class PokemonEnv:
    def __init__(self, hwnd=None, max_steps=1024, save_slot=1, coach=None, read_text=True, map_places=None,
                 visits=None):
        self.emu = MelonDS(hwnd)
        self.visits = visits or VisitCounts()  # curiosite partagee entre les fenetres (voir VisitCounts)
        self.line_novelty = {}                 # nouveaute des dernieres repliques lues
        self.map_places = map_places or []  # numero de carte -> nom du lieu (pour les logs)
        self.state = GameState(DSMemory(self.emu.hwnd))
        self.max_steps = max_steps
        self.save_slot = save_slot
        self.coach = coach  # partage entre toutes les fenetres (voir coach.py)
        # Lire les dialogues coute ~15 ms par action : sans coach, on s'en passe (et des recompenses
        # / penalites qui en dependent : dialogues nouveaux, relecture, sauvegarde)
        self.read_text = read_text
        self.texts = TextTracker()
        self.last_lines = []
        self._rect = None
        self.game_fps = 60
        self.ensure_fast_forward(True)

    # ---------- avance rapide : on lit les FPS dans le titre "[199/60] melonDS 1.1" ----------
    def fps(self):
        m = re.match(r"\[(\d+)/", self.emu.title())
        return int(m.group(1)) if m else 0

    def ensure_fast_forward(self, on):
        """On ne devine pas l'etat de l'avance rapide avec un seuil (si le processeur est charge, elle
        peut etre allumee a 70 images/s) : on essaie l'autre position et on garde la bonne."""
        time.sleep(1.1)  # le titre se met a jour une fois par seconde
        before = self.fps()
        if not on and before <= 62:
            self.game_fps = before or 60
            return  # deja a vitesse normale
        self.emu.toggle_fast_forward()
        time.sleep(2.2)
        after = self.fps()
        if (after > before) != on:  # c'etait deja dans le bon etat : on revient en arriere
            self.emu.toggle_fast_forward()
            time.sleep(2.2)
            after = self.fps()
        self.game_fps = max(after, 30)

    def _wait_frames(self, frames, minimum=0.0):
        time.sleep(max(frames / self.game_fps, minimum))

    # ---------- l'ecran du haut, d'apres la disposition de melonDS ----------
    def top_screen_rect(self, img):
        """melonDS empile les 2 ecrans (256x192 chacun) au centre, sous la barre de menu.
        La barre de menu se repere en cherchant la premiere ligne noire dans la marge de gauche."""
        h, w = img.shape[:2]
        col = img[:, 2].max(axis=1)
        menu = int(np.argmax(col < 30)) if (col < 30).any() else 25
        scale = min(w / 256, (h - menu) / 384)
        sw, sh = int(256 * scale), int(192 * scale)
        x0 = (w - sw) // 2
        y0 = menu + ((h - menu) - 2 * sh) // 2
        return x0, y0, sw, sh

    def observe(self):
        img = self.emu.screenshot()
        if self._rect is None:
            self._rect = self.top_screen_rect(img)
        x0, y0, sw, sh = self._rect
        top = img[y0:y0 + sh, x0:x0 + sw]
        self.last_color = top  # pour l'affichage (viewer.py), l'IA ne voit que la version grise
        gray = cv2.cvtColor(np.ascontiguousarray(top), cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)

    @staticmethod
    def looks_like_options(gray):
        """L'ecran ressemble-t-il a l'ecran OPTIONS de reference ?"""
        if OPTIONS_REF is None:
            return False
        return float(np.abs(gray.astype(np.int16) - OPTIONS_REF.astype(np.int16)).mean()) < OPTIONS_MAX_DIFF

    @staticmethod
    def textbox_visible(gray):
        """Boite de dialogue ouverte ? (image 96x72) Le bas de l'ecran devient presque blanc,
        alors que juste au-dessus c'est le decor. Si le decor est clair aussi (neige), on prefere
        ne rien detecter que de se tromper."""
        inside = (gray[57:68, 6:90] > 190).mean()
        above = (gray[48:52, 6:90] > 190).mean()
        return inside > 0.75 and above < 0.5

    def visit_map(self):
        """Image 96x72 centree sur le joueur (sur cette carte) :
        blanc = deja visitee dans cet episode ; gris = connue des episodes passes (plus c'est clair,
        plus elle est banale) ; noir = jamais vue. Les zones sombres sont celles qui rapportent."""
        map_id, px, py = self.state.position()
        grid = np.zeros((MAP_H, MAP_W), np.uint8)
        half_w, half_h = MAP_W // 2, MAP_H // 2
        for (x, y), n in list(self.visits.tiles.get(map_id, {}).items()):
            gx, gy = x - px + half_w, y - py + half_h
            if 0 <= gx < MAP_W and 0 <= gy < MAP_H:
                grid[gy, gx] = int(200 * (1 - 1 / math.sqrt(1 + n)))
        for x, y in self.tiles_by_map.get(map_id, ()):
            gx, gy = x - px + half_w, y - py + half_h
            if 0 <= gx < MAP_W and 0 <= gy < MAP_H:
                grid[gy, gx] = 255
        self.last_map = np.kron(grid, np.ones((2, 2), np.uint8))  # chaque case -> 2x2 pixels
        return self.last_map

    def _stacked(self):
        return np.concatenate([np.stack(self.frames), self.visit_map()[None]])  # (4, 72, 96)

    # ---------- appuyer sur une touche ----------
    def _do(self, button):
        """Appuie sur la touche. Renvoie False si c'etait une fleche qui n'a pas fait bouger (mur)."""
        if button in MOVES:
            # Un pas exactement : maintenir jusqu'a ce que la position change (ou un mur)
            p0 = self.state.position()
            t = time.perf_counter()
            timeout = STEP_TIMEOUT_FRAMES / self.game_fps
            self.emu.key(button, up=False)
            while self.state.position() == p0 and time.perf_counter() - t < timeout:
                time.sleep(0.001)  # laisse respirer les autres fenetres qui jouent en parallele
            self.emu.key(button, up=True)
            moved = self.state.position() != p0
            self._wait_frames(STEP_SETTLE_FRAMES)
            return moved
        # au moins 10 ms d'appui : melonDS ne lit le clavier qu'une fois par image
        self.emu.press(button, max(BUTTON_HOLD_FRAMES / self.game_fps, 0.01))
        self._wait_frames(BUTTON_SETTLE_FRAMES)
        return True

    # ---------- l'interface pour l'IA ----------
    def set_start(self, slot):
        """Change la sauvegarde de depart (apres une etape importante, voir train.py)."""
        self.save_slot = slot
        self.start_pos = None  # le prochain reset memorisera le nouveau point de depart

    def reset(self):
        # Le premier reset memorise le point de depart ; les suivants doivent y revenir exactement
        for attempt in range(3):
            self.emu.load_state(self.save_slot)
            self._wait_frames(18, minimum=0.15)  # le chargement lui-meme prend du temps reel
            pos = self.state.position()
            if getattr(self, "start_pos", None) in (None, pos):
                break
        else:
            print(f"[attention] le reset n'a pas marche (position {pos}, attendu {self.start_pos})")
        if getattr(self, "start_pos", None) is None:
            self.start_pos = pos
        self._rect = None
        self.steps = 0
        self.visited = set()
        self.tiles_by_map = {}  # carte -> {(x, y)} pour dessiner la carte des cases visitees
        self.maps = set()
        # Pour les logs : d'ou viennent les points, ou elle est passee, ce qu'elle a lu, les incidents
        self.places_seen = []
        self.places_visited = set()  # noms de lieux deja vus dans l'episode
        self.episode_lines = []
        self.events = {"options": 0, "sauvegarde": 0, "ko": 0, "murs": 0, "bloque": 0, "menu": 0}
        self.idle_steps = 0  # actions d'affilee sans rien de nouveau
        self.points = defaultdict(float)
        self._reward_for(self.state.position())  # la case de depart ne rapporte rien...
        self.points = defaultdict(float)         # ...donc on ne la compte pas
        self.progress = self.state.summary()     # equipe / niveaux / badges au depart
        self.best = dict(self.progress)          # on ne recompense que ce qui depasse le record
        if self.read_text:
            self.texts.reset(self.state.mem.snapshot())  # le texte deja en memoire ne compte pas
        self.text_rewards = 0
        self.got_new_text = False
        self.coach_rewarded = set()  # lieux de quete et phrases-preuves deja recompenses dans cet episode
        if self.coach and pos[0] in {**self.coach.done_place_maps(), **self.coach.active_place_maps()}:
            self.coach_rewarded.add(self.place_of(pos[0]))  # on y est deja : pas de bonus gratuit
        self.frames = [self.observe()] * FRAME_STACK
        self.in_options = False
        self.textbox_open = self.textbox_visible(self.frames[-1])
        self.dialogue_wait = 0       # >0 : une boite vient de s'ouvrir, on attend du texte nouveau
        self.dialogue_is_new = False
        self.dialogue_pos = None     # case ou la boite s'est ouverte (= quel PNJ / objet, a peu pres)
        self.repeat_counts = {}      # case -> nombre de relectures depuis cette case
        self.repeats = 0             # dialogues relus (pour les logs)
        self.coach_waiting = []      # phrases lues en RAM, en attente de voir si elles sont dans la boite
        return self._stacked()

    def _earn(self, category, value):
        """Chaque point gagne ou perdu est range dans une categorie (pour les logs)."""
        self.points[category] += value
        return value

    def _send_dialogue_to_coach(self):
        """Le coach ne doit lire QUE les dialogues : une phrase trouvee en RAM ne lui est envoyee que si
        la boite de dialogue est affichee (maintenant ou dans les DIALOGUE_GRACE actions suivantes).
        Les menus, le Sac, les noms de lieux affiches en haut de l'ecran... ne passent pas."""
        if not self.coach_waiting:
            return
        if self.textbox_visible(self.frames[-1]) or self.textbox_open:
            self.coach.add_lines([line for line, _ in self.coach_waiting])
            self.coach_waiting = []
            return
        for item in self.coach_waiting:
            item[1] -= 1
        self.coach_waiting = [item for item in self.coach_waiting if item[1] > 0]

    def _menu_and_repeat_penalty(self, ram_bytes):
        """Penalites : etre dans OPTIONS, et relire un dialogue deja lu."""
        reward = 0.0
        in_options = ram_bytes.find(OPTIONS_SIGNATURE) != -1 or self.looks_like_options(self.frames[-1])
        if in_options:
            if not self.in_options:
                self.events["options"] += 1
            reward += self._earn("options", PENALTY_OPTIONS_STEP + (PENALTY_OPTIONS if not self.in_options else 0.0))
        self.in_options = in_options

        if not self.read_text:  # sans lecture du texte, impossible de savoir si un dialogue est relu
            return reward
        visible = self.textbox_visible(self.frames[-1])
        if visible and not self.textbox_open:  # une boite de dialogue vient de s'ouvrir
            self.dialogue_wait, self.dialogue_is_new = DIALOGUE_GRACE, False
            self.dialogue_pos = self.state.position()
        self.textbox_open = visible
        if self.dialogue_wait > 0:
            self.dialogue_is_new |= self.got_new_text
            self.dialogue_wait = 0 if self.dialogue_is_new else self.dialogue_wait - 1
            if self.dialogue_wait == 0 and not self.dialogue_is_new:
                # deja lu : punition qui grandit a chaque relecture depuis la meme case
                count = self.repeat_counts.get(self.dialogue_pos, 0) + 1
                self.repeat_counts[self.dialogue_pos] = count
                reward += self._earn("relus", PENALTY_REPEAT_DIALOGUE * min(count, MAX_REPEAT_FACTOR))
                self.repeats += 1
        return reward

    def _text_and_coach_reward(self, pos):
        """Nouvelles phrases de dialogue (recompense + on les raconte au coach), et lieu cible du coach."""
        reward = 0.0
        self.ram = self.state.mem.snapshot()  # sert aussi a detecter OPTIONS
        if not self.read_text:
            self.got_new_text = False
            return reward
        lines = self.texts.new_lines(self.ram)
        self.got_new_text = bool(lines)
        saving = [l for l in lines if any(s in l.lower() for s in SAVE_TEXTS)]
        if saving:  # ces phrases-la ne rapportent rien : elles coutent
            self.events["sauvegarde"] += 1
            reward += self._earn("sauvegarde", PENALTY_SAVE)
            lines = [l for l in lines if l not in saving]
        if lines:
            self.last_lines = lines
            self.episode_lines += lines
            counted = max(min(len(lines), MAX_TEXT_REWARDS - self.text_rewards), 0)
            self.text_rewards += counted
            # nouveaute depuis le debut de l'entrainement : 1 la premiere fois, puis 1/sqrt(1 + episodes)
            self.line_novelty = {line: self.visits.novelty_line(line) for line in lines}
            for line in lines[:counted]:
                reward += self._earn("dialogues", REWARD_NEW_TEXT * self.line_novelty[line])
        if self.coach:
            # Une phrase que le coach a classee "histoire" (A = quete, S = etape) vaut plus qu'un passant
            people = self.coach.quest_people()
            for line in lines:
                # Pas de baisse ici : sans sauvegarde d'etape, l'histoire (Matt...) est a refaire a chaque episode
                if self.coach.line_class(line) in ("A", "S"):
                    reward += self._earn("histoire", REWARD_STORY_TEXT)
                # La personne de la quete parle (ex. quete "...Matt..." et la replique commence par "Matt:")
                if speaker_of(line) in people:
                    reward += self._earn("pnj_quete", REWARD_QUEST_TALK)
            # pas envoyees tout de suite : seulement si la boite de dialogue est affichee (voir step)
            self.coach_waiting += [[line, DIALOGUE_GRACE] for line in lines]
            self.coach.saw_map(pos[0])
            self.coach.saw_progress(self.progress)
            # Le lieu de la quete / etape en cours, ou d'une quete / etape deja accomplie (a refaire a chaque episode)
            place = self.place_of(pos[0])
            kind = {**self.coach.done_place_maps(), **self.coach.active_place_maps()}.get(pos[0])
            if kind and place not in self.coach_rewarded:
                self.coach_rewarded.add(place)
                reward += self._earn("coach", REWARD_QUEST if kind == "quete" else REWARD_STEP)
                self.coach.place_reached(place)
            # Une phrase qui a prouve qu'une quete / etape etait accomplie : l'etape d'histoire est refaite
            for line in lines:
                kind = self.coach.proof_kind(line)
                if kind and line not in self.coach_rewarded:
                    self.coach_rewarded.add(line)
                    reward += self._earn("coach", REWARD_QUEST if kind == "quete" else REWARD_STEP)
        return reward

    def _progress_reward(self, now):
        """Recompenses de progression : equipe, niveaux, badges, et penalite si tout le monde est K.O."""
        reward = 0.0
        for key, value, category in (("party", REWARD_NEW_POKEMON, "equipe"), ("levels", REWARD_LEVEL, "niveaux"),
                                     ("badges", REWARD_BADGE, "badges")):
            if now[key] > self.best[key]:
                reward += self._earn(category, (now[key] - self.best[key]) * value)
                self.best[key] = now[key]
        if now["party"] > 0 and now["hp"] == 0 and self.progress["hp"] > 0:
            self.events["ko"] += 1
            reward += self._earn("ko", PENALTY_BLACKOUT)
        self.progress = now
        return reward

    def _reward_for(self, pos):
        reward = 0.0
        if pos[0] not in self.maps:
            self.maps.add(pos[0])
            place = self.place_of(pos[0])
            novelty = self.visits.novelty_map(pos[0])
            if place not in self.places_visited:  # un lieu jamais vu dans l'episode : gros bonus (toujours)
                self.places_visited.add(place)
                reward += self._earn("lieux", REWARD_NEW_PLACE)
            else:                                 # une piece de plus dans un lieu connu : selon sa nouveaute
                reward += self._earn("cartes", REWARD_NEW_MAP * novelty)
            if place and (not self.places_seen or self.places_seen[-1] != place):
                self.places_seen.append(place)
        if pos not in self.visited:
            self.visited.add(pos)
            tiles = self.tiles_by_map.setdefault(pos[0], set())
            tiles.add(pos[1:])
            novelty = self.visits.novelty_tile(pos)  # rare depuis le debut de l'entrainement = payant
            reward += self._earn("cases", REWARD_NEW_TILE * novelty)
        return reward

    def place_of(self, map_id):
        return self.map_places[map_id] if map_id < len(self.map_places) else f"carte {map_id}"

    def step(self, action):
        moved = self._do(ACTIONS[action])
        self.steps += 1
        if self.steps % FPS_REFRESH_STEPS == 0:  # la vitesse du jeu varie (charge du processeur)
            self.game_fps = max(self.fps(), 30)
        pos = self.state.position()
        now = self.state.summary()
        explore = self._reward_for(pos)
        reward = explore + self._progress_reward(now) + self._text_and_coach_reward(pos)
        self.frames = self.frames[1:] + [self.observe()]
        if self.coach:
            self._send_dialogue_to_coach()
        reward += self._menu_and_repeat_penalty(self.ram.tobytes())
        if not moved:  # fleche contre un mur : action perdue
            self.events["murs"] += 1
            reward += self._earn("murs", PENALTY_BUMP)
        if ACTIONS[action] in ("X", "START") and now["party"] == 0:
            self.events["menu"] += 1
            reward += self._earn("menu", PENALTY_EARLY_MENU)

        # Bloquee ? (ni case, ni carte, ni dialogue nouveaux depuis STUCK_STEPS actions) -> fin de l'episode
        self.idle_steps = 0 if (explore > 0 or self.got_new_text) else self.idle_steps + 1
        stuck = self.idle_steps >= STUCK_STEPS
        if stuck:
            self.events["bloque"] += 1
            reward += self._earn("bloque", PENALTY_STUCK)
        in_options = OPTIONS_ENDS_EPISODE and self.in_options  # entree dans OPTIONS : fin immediate
        done = self.steps >= self.max_steps or stuck or in_options
        info = {"pos": pos, "tiles": len(self.visited), "maps": len(self.maps), "texts": self.text_rewards,
                "repeats": self.repeats, "options": self.in_options, "place": self.place_of(pos[0]),
                "points": self.points, "places": self.places_seen, "lines": self.episode_lines,
                "events": self.events, "steps": self.steps, **now}
        return self._stacked(), reward, done, info

    def close(self):
        self.ensure_fast_forward(False)
