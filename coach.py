"""Le coach : un modele de langage local (Ollama) lit l'histoire du jeu et tient a jour UN plan.

  plan = {"objectif": "Rejoindre Matt sur la Route 201 puis aller au lac",
          "lieux": ["Route 201", "Lake Verity"],
          "fini_quand": {"type": "entrer_lieu", "valeur": "Route 201"}}

Principes :
  - Il raisonne par SCENE, pas par phrase : env.py accumule tout le texte tant que la boite de dialogue
    est ouverte, et lui envoie la scene entiere quand elle est fermee depuis ~10 actions.
  - Il n'appelle le modele que sur des DECLENCHEURS : une scene jamais vue avec un personnage nomme,
    une scene pendant laquelle un flag d'evenement s'est allume, ou un lieu atteint pour la 1re fois.
    Passants, tele, panneaux, scenes deja vues : filtres en Python.
  - Un CACHE permanent (scene -> reponse) : les 4 fenetres rejouent les memes scenes, chacune ne coute
    qu'un seul appel, une fois pour toutes. Le modele peut donc etre lent (14B, en arriere-plan).
  - C'est le JEU qui valide, jamais le modele : "fini_quand" est choisi dans une liste fermee, et
    env.py le verifie dans la RAM (carte, equipe, badges, flags, qui parle).
  - Un lieu cible n'est accepte que si un mot significatif de son nom est dans le texte
    ("the lake" -> "Lake Verity" accepte ; "Route 201" doit contenir "201").
  - Le carnet (l'histoire jusqu'ici) n'est reecrit que quand un objectif est valide.

Le coach ne controle pas les touches ; il oriente l'exploration (voir les recompenses dans env.py).
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque

OLLAMA = "http://localhost:11434/api"
SAVE = "checkpoints/coach_v2.json"      # etat du coach (plan, carnet, journal)
CACHE = "checkpoints/coach_cache.json"  # scene -> reponse du modele, pour toujours
OLD_SAVE = "checkpoints/coach.json"     # ancien coach : on reprend seulement son carnet
NUM_CTX = 8192          # fenetre de contexte d'Ollama (2048 par defaut : le prompt serait coupe sans prevenir)
RECENT_SCENES = 6       # scenes importantes rappelees au modele
DONE_TYPES = ["entrer_lieu", "parler_a", "obtenir_pokemon", "badge", "nouveau_flag"]

SPEAKER = re.compile(r"^([A-Z][A-Za-z.' ]{1,15}):\s")
# Emissions de tele / radio : elles parlent a tout le monde, jamais au joueur
BROADCAST = ("MC", "Commercial", "Announcer", "Reporter")
# Mots trop generaux pour ancrer un lieu dans le texte ("route" seul ne dit pas laquelle)
GENERIC_WORDS = {"town", "city", "route", "the", "of", "mt", "area", "gate", "pokemon", "center", "house"}

PROMPT = """You are the coach of an AI that is playing Pokemon Diamond (English version).
You keep ONE plan: what the player must do next in the story, where to go, and how the game will
know it is done.

YOUR NOTEBOOK - the story so far, written by you earlier:
{story}

CURRENT PLAN: {plan}

RECENT IMPORTANT SCENES (oldest first):
{recent}

Player: {party} Pokemon in team, {badges} badges. Current place: {current}.
Places already visited: {visited}

NEW EVENT:
{event}

Does this event tell the player what to do next in the STORY (or change the current plan)?
- "new_plan": true only if it does. TV shows, small talk, tips and warnings -> false.
- "objective": what the player must do, max 15 words.
- "target_places": where the player must go, chosen ONLY from this list: {places}.
  Only places the dialogue really talks about (a vague name like "the harbor" may match a place of the
  list whose name contains "Harbor"). May be empty. Never the current place.
- "done_when": how the GAME will check that the objective is done:
    "entrer_lieu" (value = one of target_places): the player enters that place,
    "parler_a" (value = the character's name, e.g. "Matt"): the player talks with that character,
    "obtenir_pokemon": the player gets a new Pokemon,
    "badge": the player wins a badge,
    "nouveau_flag": anything new happens in the story (use it when nothing else fits).

Answer only with JSON. Example from ANOTHER game (its places are not in your list): for the scene
"Ted: I'll wait for you at the Old Pier, then we sail to Rockport Harbor!", the answer would be
{{"reasoning": "Ted will wait at the Old Pier before going to the harbor.", "new_plan": true,
  "objective": "Meet Ted at the Old Pier, then sail to the harbor", "target_places": ["Old Pier", "Rockport Harbor"],
  "done_when": {{"type": "entrer_lieu", "value": "Old Pier"}}}}"""

NOTEBOOK_PROMPT = """You are the coach of an AI that is playing Pokemon Diamond (English version).
You keep a NOTEBOOK about the story, to remember it later.

Your notebook until now:
{story}

The player just DID this (checked by the game): {done}

Important scenes since your last update (oldest first):
{recent}

Rewrite the notebook in max 120 words: what happened so far, who is who, where the characters went,
what the player has done. Only facts from the dialogue: never invent anything.

Answer only with JSON: {{"notebook": "..."}}"""


def speaker_of(line):
    """Dans Diamant, les personnages importants ont leur nom devant la replique ("Mom: ...", "Rowan: ...").
    Les passants, panneaux et objets n'en ont pas."""
    m = SPEAKER.match(line)
    return m.group(1) if m else None


def _norm(text):
    """Pour comparer : minuscules, sans ponctuation ni espaces en trop."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _scene_key(lines):
    return hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()


class Coach:
    def __init__(self, map_places, model="qwen2.5:14b"):
        self.map_places = map_places                  # numero de carte -> nom du lieu
        self.places = sorted(set(map_places) - {"Mystery Zone", ""})
        self.model = self._available(model)
        self.plan = None                              # le plan en cours (voir le haut du fichier)
        self.story = ""                               # son carnet : l'histoire jusqu'ici, ecrite par lui
        self.recent = deque(maxlen=RECENT_SCENES)     # textes des scenes importantes
        self.speakers = set()                         # personnages deja entendus ("Mom", "Matt"...)
        self.visited = set()                          # lieux deja atteints (toutes fenetres)
        self.cache = {}                               # cle de scene / de lieu -> reponse du modele
        self.queued = set()                           # cles deja en attente ou traitees
        self.pending = deque()                        # travaux pour le modele (en arriere-plan)
        self.current = "?"
        self.status = {"party": 0, "badges": 0}
        self.log = []                                 # ce que le coach a decide (sauvegarde)
        self.busy = False
        self.disabled = False      # Ollama injoignable : on reessaie plus tard
        self.retry_at = 0.0
        self._lock = threading.Lock()
        self._load()

    def _available(self, model):
        """Le modele demande est-il installe ? Sinon on prend le plus gros qwen2.5 present."""
        try:
            names = [m["name"] for m in json.load(urllib.request.urlopen(f"{OLLAMA}/tags", timeout=5))["models"]]
        except Exception:
            return model  # Ollama pas lance : on verra plus tard
        if model in names:
            return model
        fallback = next((n for n in ("qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b") if n in names), model)
        print(f"[coach] {model} n'est pas installe (ollama pull {model}) : j'utilise {fallback}")
        return fallback

    # ---------- ce que le reste du programme lit ----------
    @property
    def objective(self):
        return self.plan["objectif"] if self.plan else None

    @property
    def targets(self):
        return list(self.plan["lieux"]) if self.plan else []

    def target_maps(self):
        """Numeros de carte des lieux cibles du plan en cours."""
        targets = set(self.targets)
        return {i for i, p in enumerate(self.map_places) if p in targets}

    def done_text(self):
        if not self.plan:
            return ""
        d = self.plan["fini_quand"]
        return d["type"] + (f" {d['valeur']}" if d.get("valeur") else "")

    # ---------- ce que les fenetres lui racontent ----------
    def add_scene(self, lines, ctx):
        """Une scene FINIE (toutes ses phrases, dans l'ordre). ctx = etat de la fenetre :
        party, badges, flags_total, place, new_flag (un flag s'est allume pendant la scene)."""
        with self._lock:
            lines = [l for l in lines if len(l) >= 12 and l not in self.places
                     and not (speaker_of(l) or "").startswith(BROADCAST)]
            named = {s for s in map(speaker_of, lines) if s}
            self.speakers |= named
            self._update_status(ctx)
            if not lines:
                return
            # Le jeu valide "parler_a" : une scene (autre que celle qui a donne le plan) ou il parle
            if self.plan and self.plan["fini_quand"]["type"] == "parler_a":
                who = self.plan["fini_quand"]["valeur"].lower()
                if any(s.lower() == who for s in named) and _scene_key(lines) != self.plan.get("scene"):
                    self._complete(f"{self.plan['fini_quand']['valeur']} a parle")
            key = _scene_key(lines)
            if key in self.queued:
                return  # deja vue (les 4 fenetres refont les memes scenes)
            self.queued.add(key)
            if not named and not ctx.get("new_flag"):
                return  # passant, panneau, objet, sans evenement : pas la peine de demander
            event = "A new scene of dialogue (complete, in order; \"Name:\" = who speaks):\n" + \
                    "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1))
            if ctx.get("new_flag"):
                event += "\n(The game registered a new story event during this scene.)"
            self.pending.append({"key": key, "event": event, "text": " ".join(lines), "ctx": dict(ctx)})

    def saw_place(self, place, ctx):
        """La fenetre est sur une carte de ce lieu. 1re fois qu'un lieu est atteint : declencheur."""
        with self._lock:
            self.current = place
            self._update_status(ctx)
            if not place or place == "Mystery Zone" or place in self.visited:
                return
            self.visited.add(place)
            key = f"lieu:{place}"
            if key in self.queued:
                return
            self.queued.add(key)
            self.pending.append({"key": key, "event": f"The player arrived for the first time at: {place}.",
                                 "text": place, "ctx": dict(ctx)})

    def check(self, place, party, badges, flags_total):
        """Appele a chaque action par chaque fenetre : le JEU dit si le plan est accompli."""
        plan = self.plan
        if not plan:
            return
        kind, value, base = plan["fini_quand"]["type"], plan["fini_quand"]["valeur"], plan["base"]
        done = ((kind == "entrer_lieu" and place == value)
                or (kind == "obtenir_pokemon" and party > base["party"])
                or (kind == "badge" and badges > base["badges"])
                or (kind == "nouveau_flag" and flags_total > base["flags_total"]))
        if done:
            with self._lock:
                if self.plan is plan:  # (une autre fenetre a pu le valider juste avant)
                    self._complete(f"{kind} {value}".strip())

    def _update_status(self, ctx):
        for key in ("party", "badges"):
            self.status[key] = max(self.status[key], ctx.get(key, 0))

    # ---------- la question au modele, en arriere-plan ----------
    def maybe_ask(self):
        if self.disabled and time.time() >= self.retry_at:
            self.disabled = False  # on retente Ollama
        if self.pending and not self.busy and not self.disabled:
            self.busy = True
            threading.Thread(target=self._work, args=(self.pending.popleft(),), daemon=True).start()

    def _query(self, prompt, schema):
        body = {"model": self.model, "prompt": prompt, "format": schema, "stream": False,
                "options": {"temperature": 0, "num_ctx": NUM_CTX}}
        req = urllib.request.Request(f"{OLLAMA}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(json.load(urllib.request.urlopen(req, timeout=600))["response"])

    def _plan_schema(self):
        """Sortie imposee par schema JSON (Ollama) : les lieux et les types de fin sont des listes fermees."""
        return {"type": "object", "required": ["reasoning", "new_plan", "objective", "target_places", "done_when"],
                "properties": {
                    "reasoning": {"type": "string"},
                    "new_plan": {"type": "boolean"},
                    "objective": {"type": "string"},
                    "target_places": {"type": "array", "items": {"type": "string", "enum": self.places}},
                    "done_when": {"type": "object", "required": ["type", "value"],
                                  "properties": {"type": {"type": "string", "enum": DONE_TYPES},
                                                 "value": {"type": "string"}}}}}

    def _describe_plan(self):
        if not self.plan:
            return "none yet"
        return f"{self.plan['objectif']} (places: {', '.join(self.plan['lieux']) or 'none'}; " \
               f"done when: {self.done_text()})"

    def _work(self, job):
        try:
            if job.get("notebook"):
                self._update_notebook(job["done"])
                return
            answer = self.cache.get(job["key"])
            if answer is None:
                prompt = PROMPT.format(
                    story=self.story or "(empty: the game just started)", plan=self._describe_plan(),
                    recent="\n".join(f"- {s}" for s in self.recent) or "- (none yet)",
                    party=self.status["party"], badges=self.status["badges"], current=self.current,
                    visited=", ".join(sorted(self.visited)) or "none", event=job["event"],
                    places=", ".join(self.places))
                answer = self._query(prompt, self._plan_schema())
                with self._lock:
                    self.cache[job["key"]] = answer
                    self._save_cache()
            with self._lock:
                self._apply(job, answer)
                self._save()
        except urllib.error.URLError as e:
            print(f"[coach] Ollama injoignable ({e}) : nouvel essai dans 60 s, l'entrainement continue")
            self.pending.appendleft(job)  # le travail n'est pas perdu
            self.disabled, self.retry_at = True, time.time() + 60
        except Exception as e:  # reponse mal formee : on passe au suivant
            print(f"[coach] erreur : {e}")
        finally:
            self.busy = False

    def _anchored(self, place, text):
        """Un mot significatif du nom du lieu est-il dans le texte ? ("lake" pour "Lake Verity")."""
        words = set(_norm(text).split())
        return any(w in words for w in _norm(place).split() if w not in GENERIC_WORDS)

    def _apply(self, job, answer):
        if job["text"] and not job["key"].startswith("lieu:"):
            self.recent.append(job["text"][:400])
        if not answer.get("new_plan"):
            return
        objective = str(answer.get("objective") or "").strip()[:120]
        if not objective:
            return
        # Lieux ancres dans le texte : la scene, ou les scenes importantes recentes (le plan peut les reprendre)
        context = job["text"] + " " + " ".join(self.recent)
        places, refused = [], []
        for p in answer.get("target_places") or []:
            if p in self.places and p != self.current and p not in places:
                (places if self._anchored(p, context) else refused).append(p)
        if refused:
            print(f"[coach] lieux refuses (pas dans le texte) : {', '.join(refused)}")
        done = answer.get("done_when") or {}
        kind, value = str(done.get("type") or ""), str(done.get("value") or "").strip()
        if kind == "entrer_lieu" and value not in places:
            kind, value = ("entrer_lieu", places[0]) if places else ("nouveau_flag", "")
        if kind == "parler_a":
            match = next((s for s in self.speakers if s.lower() == value.lower()), None)
            kind, value = ("parler_a", match) if match else ("nouveau_flag", "")
        if kind not in DONE_TYPES:
            kind, value = "nouveau_flag", ""
        if kind in ("obtenir_pokemon", "badge", "nouveau_flag"):
            value = ""
        ctx = job["ctx"]
        self.plan = {"objectif": objective, "lieux": places, "fini_quand": {"type": kind, "valeur": value},
                     "base": {"party": ctx.get("party", 0), "badges": ctx.get("badges", 0),
                              "flags_total": ctx.get("flags_total", 0)},
                     "scene": job["key"], "depuis": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._note("nouveau plan", f"{objective} | lieux : {', '.join(places) or '-'} | fini quand : {self.done_text()}",
                   answer.get("reasoning"))
        print(f"[coach] ({answer.get('reasoning')})\n[coach] nouveau plan : {objective}"
              f"  | lieux : {', '.join(places) or '-'} | fini quand : {self.done_text()}")

    def _complete(self, why):
        """Le jeu a valide le plan : on le note, et le carnet est reecrit (seulement a ce moment-la)."""
        done = self.plan["objectif"]
        self._note("plan accompli", done, why)
        print(f"[coach] plan accompli : {done} ({why})")
        self.plan = None
        self.pending.append({"notebook": True, "done": f"{done} ({why})", "key": None})
        self._save()

    def _update_notebook(self, done):
        prompt = NOTEBOOK_PROMPT.format(story=self.story or "(empty)", done=done,
                                        recent="\n".join(f"- {s}" for s in self.recent) or "- (none)")
        notebook = str(self._query(prompt, "json").get("notebook") or "").strip()
        if notebook:
            with self._lock:
                self.story = notebook[:1200]
                self._save()
            print(f"[coach] carnet mis a jour : {self.story}")

    # ---------- sauvegarde (on reprend la ou on en etait au prochain lancement) ----------
    def _note(self, event, what, detail):
        self.log.append({"heure": time.strftime("%Y-%m-%d %H:%M:%S"), "evenement": event,
                         "objectif": what, "detail": detail})

    def save(self):
        """Appele par train.py a chaque sauvegarde du cerveau (garde aussi les travaux en attente)."""
        with self._lock:
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(SAVE), exist_ok=True)
        json.dump({"plan": self.plan, "story": self.story, "recent": list(self.recent),
                   "speakers": sorted(self.speakers), "visited": sorted(self.visited),
                   "queued": sorted(k for k in self.queued if k), "pending": list(self.pending),
                   "log": self.log[-200:]},
                  open(SAVE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def _save_cache(self):
        tmp = CACHE + ".tmp"
        json.dump(self.cache, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        os.replace(tmp, CACHE)

    def _load(self):
        if os.path.exists(CACHE):
            self.cache = json.load(open(CACHE, encoding="utf-8"))
        if os.path.exists(SAVE):
            saved = json.load(open(SAVE, encoding="utf-8"))
            self.plan = saved.get("plan")
            self.story = saved.get("story", "")
            self.recent.extend(saved.get("recent", []))
            self.speakers = set(saved.get("speakers", []))
            self.visited = set(saved.get("visited", []))
            self.queued = set(saved.get("queued", []))
            self.pending.extend(saved.get("pending", []))  # travaux pas encore faits au dernier arret
            self.log = saved.get("log", [])
        elif os.path.exists(OLD_SAVE):  # ancien coach (quete / etape) : on garde son carnet et ses personnages
            old = json.load(open(OLD_SAVE, encoding="utf-8"))
            self.story = old.get("story", "")
            self.speakers = set(old.get("speakers", []))
