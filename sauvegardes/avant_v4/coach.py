"""Le coach : un modele de langage local (Ollama) lit les dialogues du jeu et tient a jour deux objectifs.

  - la QUETE principale (A) : l'objectif d'histoire a long terme ("aller au lac avec Matt").
    Seul un personnage NOMME ("Mom: ...", "Matt: ...") peut la donner.
  - l'ETAPE (S) : la prochaine chose a faire tout de suite ("aller voir chez Matt").
    N'importe qui peut l'indiquer, meme un passant. Une etape ne remplace jamais la quete.
  - B = conseil (commandes, avertissements) et C = inutile : ignores.

Pour chaque nouveau dialogue, UN appel au modele qui classe la phrase et dit si elle prouve que la
quete ou l'etape est accomplie. Il doit CITER la preuve, et on verifie que la citation est vraiment
dans la phrase lue (sinon c'est refuse). Arriver au lieu d'une quete / etape l'accomplit aussi.
Un lieu n'est accepte que s'il est ecrit dans la replique : le modele ne doit jamais deviner.

Les recompenses sont donnees par env.py une fois par episode et par fenetre (lieu atteint, ou phrase-
preuve relue = l'etape d'histoire refaite). Le coach ne controle pas les touches, et il tourne en
arriere-plan pour ne pas ralentir le jeu.
"""
import difflib
import json
import os
import re
import threading
import time
import urllib.request
from collections import deque

OLLAMA = "http://localhost:11434/api/generate"
SAVE = "checkpoints/coach.json"
QUEST, STEP = "quete", "etape"

PROMPT = """You are the coach of an AI that is playing Pokemon Diamond (English version).
You read the game's dialogue and keep track of two things:
- the MAIN QUEST: the long-term story objective;
- the NEXT STEP: the very next thing the player should do right now.

YOUR NOTEBOOK - the story so far, written by you earlier:
{story}

YOUR PAST DECISIONS (oldest first):
{decisions}

Main quest: {quest}
Next step: {step}
Recent important dialogue (oldest first):
{history}

Player: {party} Pokemon in team, {badges} badges. Current place: {current}.
Places already visited: {visited}
Places that exist in the game: {places}

NEW dialogue line, said by {speaker}:
"{line}"

Tasks:
1. Classify the NEW line:
   "A" = a NAMED character gives a new long-term story objective,
   "S" = someone tells the player where to go or what to do NEXT, right now (a next step),
   "B" = a tip, a warning or general advice (controls, items, menus, "be careful...", "don't go..."),
   "C" = useless (small talk, signs, flavour text).
   Examples (not from this game):
   - "Ted: Please bring this letter to my brother in Rockport City." -> A
   - "Your sister went back to the harbor. Why don't you go and see her?" -> S
   - "Ted: Oops, I left my map at home! Wait here!" -> S (the next step changes)
   - "Be careful, wild Pokemon live in tall grass." -> B (a warning, not a quest)
   - "Press START to see the map." -> B
   - "What a nice day for a walk!" -> C
   TV and radio shows (an MC, a reporter, a commercial...) talk to everybody, not to the player:
   they are always "C".
   When unsure, prefer B or C: objectives must never be replaced by mere advice.
2. If "A" or "S": "objective" is REQUIRED (never null): what the player must do, in max 12 words,
   for example "Go see the sister at the harbor". Set "place" ONLY if the place
   name is written in the NEW line (copied exactly from "Places that exist in the game"), else null.
   Never guess a place, and never write a place name in "objective" that is not in the NEW line.
3. Does the NEW line show that the main quest or the next step is accomplished? Only true if the line
   shows the task was DONE (the player met the person, delivered the item, arrived, won...). A line
   that only mentions the same person or place is NOT a proof. If yes, "proof" MUST be copied word for
   word from the NEW line.

Answer only with JSON, like this example (for the line "Your sister went back to the harbor. Why don't you
go and see her?"):
{{"reasoning": "Someone tells the player to go and see the sister.", "class": "S",
  "objective": "Go and see the sister at the harbor", "place": null,
  "quest_done": false, "step_done": false, "proof": null}}"""


NOTEBOOK_PROMPT = """You are the coach of an AI that is playing Pokemon Diamond (English version).
You keep a NOTEBOOK about the story, to remember it later.

Your notebook until now:
{story}

New important dialogue since your last update (oldest first):
{lines}

Main quest now: {quest}
Next step now: {step}

Rewrite the notebook in max 120 words: what happened so far, who is who, where the characters went,
what the player has done and must do next. Only facts from the dialogue: never invent anything.

Answer only with JSON: {{"notebook": "..."}}"""

NOTEBOOK_EVERY = 3   # le carnet est reecrit tous les 3 dialogues importants (ou quand un objectif change)
NUM_CTX = 8192       # fenetre de contexte d'Ollama (2048 par defaut : le prompt serait coupe sans prevenir)


SPEAKER = re.compile(r"^([A-Z][A-Za-z.' ]{1,15}):\s")
# Emissions de tele / radio : elles parlent a tout le monde, jamais au joueur -> jamais un objectif
BROADCAST = ("MC", "Commercial", "Announcer", "Reporter")


def speaker_of(line):
    """Dans Diamant, les personnages importants ont leur nom devant la replique ("Mom: ...", "Rowan: ...").
    Les passants, panneaux et objets n'en ont pas."""
    m = SPEAKER.match(line)
    return m.group(1) if m else None


def _norm(text):
    """Pour comparer une citation a la phrase : minuscules, sans ponctuation ni espaces en trop."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _empty():
    return {"id": 0, "texte": None, "lieu": None, "active": False}


class Coach:
    def __init__(self, map_places, model="qwen2.5:7b"):
        self.map_places = map_places                  # numero de carte -> nom du lieu
        self.places = sorted(set(map_places) - {"Mystery Zone"})
        self.model = model
        self.goals = {QUEST: _empty(), STEP: _empty()}
        self.history = deque(maxlen=30)               # dialogues classes A, S ou B (contexte du modele)
        self.story = ""                               # son carnet : l'histoire jusqu'ici, ecrite par lui
        self.unnoted = []                             # dialogues importants pas encore dans le carnet
        self.proofs = {}                              # phrase-preuve -> "quete" ou "etape"
        self.done_places = {}                         # lieu accompli -> "quete" ou "etape"
        self.seen = set()                             # phrases deja envoyees au modele
        self.classes = {}                             # phrase -> classe A / S / B / C (bonus "histoire")
        self.speakers = set()                         # personnages deja entendus ("Mom", "Matt"...)
        self.pending = deque()                        # phrases en attente d'etre classees
        self.visited = set()
        self.current = "?"
        self.status = {"party": 0, "badges": 0}
        self.log = []                                 # ce que le coach a decide (sauvegarde)
        self.busy = False
        self.disabled = False      # Ollama injoignable : on reessaie plus tard
        self.retry_at = 0.0
        self._lock = threading.Lock()
        self._load()

    # ---------- ce que le reste du programme lit ----------
    def text(self, kind):
        return self.goals[kind]["texte"] if self.goals[kind]["active"] else None

    def place(self, kind):
        return self.goals[kind]["lieu"] if self.goals[kind]["active"] else None

    @property
    def objective(self):  # la quete principale (affichage, logs)
        return self.text(QUEST)

    @property
    def target(self):
        return self.place(QUEST)

    def active_place_maps(self):
        """{numero de carte: "quete"/"etape"} pour les lieux de la quete et de l'etape en cours."""
        return self._maps_of({self.place(k): k for k in (QUEST, STEP) if self.place(k)})

    def done_place_maps(self):
        return self._maps_of(self.done_places)

    def quest_people(self):
        """Les personnages (deja entendus dans le jeu) nommes dans la quete ou l'etape en cours."""
        words = set(re.findall(r"[A-Za-z]+", " ".join(t for t in (self.text(QUEST), self.text(STEP)) if t)))
        return {s for s in self.speakers if s in words}

    def line_class(self, line):
        """Le classement que le coach a donne a cette phrase (A, S, B, C), ou None s'il ne l'a pas encore vue."""
        return self.classes.get(line)

    def proof_kind(self, line):
        """'quete' / 'etape' si cette phrase a prouve une quete / etape accomplie, sinon None."""
        return self.proofs.get(line)

    # ---------- ce que les fenetres lui racontent ----------
    def add_lines(self, lines):
        with self._lock:
            for line in lines:
                speaker = speaker_of(line)
                if speaker and not speaker.startswith(BROADCAST):
                    self.speakers.add(speaker)
                if line in self.seen or line in self.places or len(line) < 15:
                    continue  # doublon, nom de lieu (panneau), ou trop court pour etre utile
                if (speaker_of(line) or "").startswith(BROADCAST):
                    self.seen.add(line)
                    self.classes[line] = "C"
                    continue  # tele / radio : pas la peine de demander au modele
                self.seen.add(line)
                self.pending.append(line)

    def saw_map(self, map_id):
        place = self.map_places[map_id] if map_id < len(self.map_places) else "?"
        self.current = place
        if place != "Mystery Zone":
            self.visited.add(place)

    def saw_progress(self, summary):
        for key in ("party", "badges"):
            self.status[key] = max(self.status[key], summary[key])

    def place_reached(self, place):
        """Une fenetre est arrivee au lieu de la quete / de l'etape : accomplie (mesure fiable)."""
        with self._lock:
            for kind in (QUEST, STEP):
                if self.place(kind) == place:
                    self.done_places[place] = kind
                    self._finish(kind, f"arrivee a {place}")

    # ---------- la question au modele, en arriere-plan ----------
    def maybe_ask(self):
        if self.disabled and time.time() >= self.retry_at:
            self.disabled = False  # on retente Ollama
        if self.pending and not self.busy and not self.disabled:
            self.busy = True
            threading.Thread(target=self._ask, args=(self.pending.popleft(),), daemon=True).start()

    def _describe(self, kind):
        text = self.text(kind)
        if not text:
            return "none yet"
        return text + (f" (place: {self.place(kind)})" if self.place(kind) else "")

    def _query(self, prompt):
        body = json.dumps({"model": self.model, "prompt": prompt, "format": "json", "stream": False,
                           "options": {"temperature": 0.1, "num_ctx": NUM_CTX}}).encode()
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        return json.loads(json.load(urllib.request.urlopen(req, timeout=180))["response"])

    def _decisions(self):
        """Ses decisions passees, pour qu'il reste coherent avec lui-meme."""
        lines = []
        for e in self.log[-15:]:
            lines.append(f"- {e['evenement']} : {e['objectif']}  (because: {str(e['detail'])[:90]})")
        return "\n".join(lines) or "- (none yet)"

    def _ask(self, line):
        try:
            prompt = PROMPT.format(
                story=self.story or "(empty: the game just started)", decisions=self._decisions(),
                quest=self._describe(QUEST), step=self._describe(STEP),
                history="\n".join(f"- {l}" for l in self.history) or "- (none yet)",
                party=self.status["party"], badges=self.status["badges"], current=self.current,
                visited=", ".join(sorted(self.visited)) or "none", places=", ".join(self.places), line=line,
                speaker=speaker_of(line) or "nobody important (a passer-by, a sign or an object)")
            answer = self._query(prompt)
            with self._lock:
                changed = self._apply(line, answer)
                self._save()
            # Le carnet : reecrit tous les NOTEBOOK_EVERY dialogues importants, ou quand un objectif change
            if self.unnoted and (changed or len(self.unnoted) >= NOTEBOOK_EVERY):
                self._update_notebook()
        except urllib.error.URLError as e:
            print(f"[coach] Ollama injoignable ({e}) : nouvel essai dans 60 s, l'entrainement continue")
            self.pending.appendleft(line)  # la phrase n'est pas perdue
            self.disabled, self.retry_at = True, time.time() + 60
        except Exception as e:  # reponse mal formee : la phrase est perdue, on passe a la suivante
            print(f"[coach] erreur : {e}")
        finally:
            self.busy = False

    def _update_notebook(self):
        prompt = NOTEBOOK_PROMPT.format(
            story=self.story or "(empty)", lines="\n".join(f"- {l}" for l in self.unnoted),
            quest=self._describe(QUEST), step=self._describe(STEP))
        notebook = str(self._query(prompt).get("notebook") or "").strip()
        if notebook:
            with self._lock:
                self.story, self.unnoted = notebook[:1200], []
                self._save()
            print(f"[coach] carnet mis a jour : {self.story}")

    def _apply(self, line, answer):
        kind = str(answer.get("class", "C")).strip().upper()[:1]
        if kind == "A" and not speaker_of(line):
            # Seul un personnage nomme donne une quete principale : un passant, au mieux une etape
            print(f"[coach] pas de quete principale sans personnage nomme -> etape : \"{line[:60]}\"")
            kind = "S"
        self.classes[line] = kind if kind in ("A", "S", "B") else "C"
        if kind in ("A", "S", "B"):
            self.history.append(line)
            self.unnoted.append(line)
        changed = False

        # 1) Quete / etape accomplie ? Seulement avec une vraie citation de la phrase.
        for done_key, goal in (("quest_done", QUEST), ("step_done", STEP)):
            if answer.get(done_key) and self.goals[goal]["active"]:
                proof = str(answer.get("proof") or "")
                if len(_norm(proof)) >= 8 and _norm(proof) in _norm(line):
                    self.proofs[line] = goal
                    self._finish(goal, f"preuve : \"{proof}\"")
                    changed = True
                else:
                    print(f"[coach] '{goal} accomplie' refusee : la preuve \"{proof}\" n'est pas dans la phrase")

        # 2) Nouvel objectif ? A -> quete principale (et l'etape repart de zero), S -> etape seulement.
        objective = str(answer.get("objective") or answer.get("new_quest") or "").strip()
        if kind in ("A", "S") and objective:
            goal = QUEST if kind == "A" else STEP
            place = self._match(str(answer.get("place") or ""))
            if place == self.current:  # deja sur place : ce lieu ne prouverait rien (bonus gratuit)
                place = None
            if place and _norm(place) not in _norm(line):  # pas ecrit dans la replique : devine, refuse
                print(f"[coach] lieu \"{place}\" refuse : il n'est pas ecrit dans la replique")
                place = None
            for name in self.places:  # le texte ne doit pas non plus citer un lieu devine
                if name in objective and _norm(name) not in _norm(line):
                    objective = objective.replace(name, "the place mentioned")
            self.goals[goal] = {"id": self.goals[goal]["id"] + 1, "texte": objective[:100], "lieu": place,
                                "active": True}
            if goal == QUEST:
                self.goals[STEP] = dict(_empty(), id=self.goals[STEP]["id"])  # nouvelle quete : etape a refaire
            label = "nouvelle quete" if goal == QUEST else "nouvelle etape"
            self._note(label, objective + (f" -> {place}" if place else ""), line)
            print(f"[coach] ({answer.get('reasoning')})\n[coach] {label} : {objective}"
                  + (f"  ->  lieu : {place}" if place else ""))
            changed = True
        return changed

    def _finish(self, kind, why):
        self.goals[kind]["active"] = False
        self._note(f"{kind} accomplie", self.goals[kind]["texte"], why)
        print(f"[coach] {kind} accomplie : {self.goals[kind]['texte']} ({why})")
        self._save()

    def _match(self, name):
        """Le modele peut mal recopier un nom de lieu : on prend le plus proche, s'il est assez proche."""
        found = difflib.get_close_matches(name, self.places, n=1, cutoff=0.75) if name else []
        return found[0] if found else None

    def _maps_of(self, place_kinds):
        return {i: place_kinds[p] for i, p in enumerate(self.map_places) if p in place_kinds}

    # ---------- sauvegarde (on reprend la ou on en etait au prochain lancement) ----------
    def _note(self, event, what, detail):
        self.log.append({"heure": time.strftime("%Y-%m-%d %H:%M:%S"), "evenement": event,
                         "objectif": what, "detail": detail})

    def _save(self):
        os.makedirs(os.path.dirname(SAVE), exist_ok=True)
        json.dump({"quest": self.goals[QUEST], "step": self.goals[STEP], "story": self.story,
                   "unnoted": self.unnoted, "history": list(self.history),
                   "proofs": self.proofs, "done_places": self.done_places, "classes": self.classes,
                   "speakers": sorted(self.speakers),
                   "seen": sorted(self.seen),
                   "visited": sorted(self.visited), "log": self.log[-200:]},
                  open(SAVE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def _load(self):
        if not os.path.exists(SAVE):
            return
        saved = json.load(open(SAVE, encoding="utf-8"))
        if "step" not in saved:  # ancien format du coach : on repart de zero
            return
        self.goals = {QUEST: saved["quest"], STEP: saved["step"]}
        self.story = saved.get("story", "")
        self.unnoted = saved.get("unnoted", [])
        self.history.extend(saved.get("history", []))
        self.proofs = dict(saved.get("proofs", {}))
        self.done_places = dict(saved.get("done_places", {}))
        self.classes = dict(saved.get("classes", {}))
        self.speakers = set(saved.get("speakers", []))
        self.seen = set(saved.get("seen", []))
        self.visited = set(saved.get("visited", []))
        self.log = saved.get("log", [])
        for line in self.seen:  # les anciennes sauvegardes n'avaient pas la liste : on la reconstruit
            speaker = speaker_of(line)
            if speaker and not speaker.startswith(BROADCAST):
                self.speakers.add(speaker)
