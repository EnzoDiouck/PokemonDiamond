"""Entrainement : l'IA joue dans toutes les fenetres melonDS ouvertes, et s'ameliore.

Lancer :  python train.py   (apres avoir ouvert 1 ou plusieurs melonDS avec la ROM et le jeu en cours)
          python train.py --voir                  -> tableau de bord : une case par fenetre DS
          python train.py --voir --capture --fenetre 2  -> idem + l'ecran du jeu de la fenetre 2
          python train.py --video demo.mp4 --fenetre 2  -> video de la case de la fenetre 2
          python train.py --sans-coach            -> sans le coach Ollama (les dialogues restent lus)
          python train.py --sans-dialogues        -> sans lecture des dialogues (plus rapide, mais
                                                     plus de recompense pour parler ni de punition si relu)
Arreter : Ctrl+C (le cerveau est sauvegarde, on reprend la ou on en etait au prochain lancement)
Les fenetres peuvent etre cachees derriere d'autres : tu peux utiliser ton PC pendant ce temps.

Logs : logs/episodes_v3.csv (un episode par ligne : date, session, points par categorie, lieux, dialogues...)
"""
import argparse
import faulthandler
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from env import ACTIONS, OBS_CHANNELS, PokemonEnv, VisitCounts
from logbook import append_csv, now
from melonds import find_windows, is_frozen, kill_window_process, launch_instance
from memory import DSMemory
from ppo import PPO
from rom_data import map_places

parser = argparse.ArgumentParser()
parser.add_argument("--voir", action="store_true", help="tableau de bord en direct (une case par fenetre DS)")
parser.add_argument("--capture", action="store_true", help="ajouter l'ecran du jeu de la fenetre choisie")
parser.add_argument("--video", help="filmer la case de la fenetre choisie dans ce fichier .mp4")
parser.add_argument("--fenetre", type=int, default=1, help="fenetre mise en avant : capture et video (1, 2...)")
parser.add_argument("--sans-coach", action="store_true", help="ne pas utiliser le coach Ollama")
parser.add_argument("--sans-dialogues", action="store_true",
                    help="ne pas lire les dialogues (plus rapide ; le coach est alors coupe aussi)")
parser.add_argument("--modele", default="qwen2.5:7b", help="modele Ollama du coach (ex. qwen2.5:3b, plus leger)")
args = parser.parse_args()

ROLLOUT = 128              # actions jouees PAR FENETRE avant chaque mise a jour du cerveau
# v3 = images + carte des cases visitees, 8 touches (les versions 1 et 2 ont ete supprimees)
CHECKPOINT = "checkpoints/ppo_v3.pt"
LOG = "logs/episodes_v3.csv"
PROGRESS = "checkpoints/progression.json"
VISITS = "checkpoints/visites.json"  # combien d'episodes sont passes par chaque case (curiosite)
SESSION = time.strftime("%Y-%m-%d_%H-%M")  # identifiant de ce lancement (retrouve dans chaque episode)

ROM_DIR = r"C:\Users\enzop\Desktop\Pokemon - Diamond Version (Europe) (Rev 5)"
ROM_NAME = "Pokemon - Diamond Version (Europe) (Rev 5)"
FIRST_SLOT, LAST_SLOT = 2, 7  # emplacements des sauvegardes d'etapes (1 = depart, 8 = libre)

# Sortie de la maison : une fois que l'IA sort tout le temps, on repart de dehors (sauvegarde 1).
EXIT_MAP = 411         # Twinleaf Town, dehors (412 a 417 = l'interieur des maisons)
EXIT_CALM_STEPS = 15   # dehors depuis 15 actions sans dialogue ni OPTIONS : les scenes de la sortie sont passees
EXIT_SLOT = 1          # remplace la chambre (gardee dans sauvegardes/chambre.ml1)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("sauvegardes", exist_ok=True)

# Mouchard : si la boucle ne progresse plus pendant 5 min, Python ecrit ou chaque fil est bloque
# dans logs/blocage.txt (fichier vide tant que tout va bien).
BLOCAGE = open("logs/blocage.txt", "w")


def watchdog():
    faulthandler.dump_traceback_later(300, file=BLOCAGE)  # rappeler la fonction repousse l'alarme

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = PPO(len(ACTIONS), OBS_CHANNELS, device)
total_steps, episode = 0, 0
source = CHECKPOINT if os.path.exists(CHECKPOINT) else None
if source:
    ckpt = agent.load(source)
    total_steps, episode = ckpt["total_steps"], ckpt["episode"]
    print(f"Reprise depuis {source} : {total_steps} actions deja jouees, episode {episode}")

# ---------- etapes : record actuel et sauvegarde de depart ----------
progress = {"slot": 1, "record": None, "history": []}
if os.path.exists(PROGRESS):
    progress = json.load(open(PROGRESS))
    print(f"Depart depuis la sauvegarde {progress['slot']} (record : {progress['record']})")


def milestone_key(info):
    """Ce qu'on considere comme 'avancer' : d'abord les badges, puis la taille de l'equipe."""
    return [info["badges"], info["party"]]

windows = find_windows()
if not windows:
    raise SystemExit("Aucune fenetre melonDS trouvee : lance melonDS avec la ROM d'abord.")
places = map_places(DSMemory(windows[0][0]))  # numero de carte -> nom du lieu (ROM)
coach = None
if not args.sans_coach and not args.sans_dialogues:  # le coach a besoin des dialogues
    from coach import Coach
    coach = Coach(places, model=args.modele)
    print(f"Coach : {args.modele} via Ollama (objectif actuel : {coach.objective or 'aucun'})")

N = len(windows)
pool = ThreadPoolExecutor(N)  # chaque fenetre joue dans son propre fil d'execution
# Creees en parallele : chacune allume son avance rapide en ~4 s
visits = VisitCounts(VISITS)  # curiosite commune a toutes les fenetres, gardee entre les lancements
print(f"Curiosite : {sum(len(t) for t in visits.tiles.values())} cases deja connues")
envs = list(pool.map(lambda w: PokemonEnv(w[0], max_steps=1024, save_slot=progress["slot"], coach=coach,
                                          read_text=not args.sans_dialogues, map_places=places,
                                          visits=visits), windows))


def check_milestone(i, info):
    """Si la fenetre i a battu le record : sauvegarde rapide, et tout le monde repartira de la."""
    key = milestone_key(info)
    if key <= progress["record"]:
        return
    n = len(progress["history"]) + 1
    slot = min(FIRST_SLOT + n - 1, LAST_SLOT)
    envs[i].emu.save_state(slot)
    time.sleep(1.0)  # laisse melonDS ecrire le fichier
    label = f"etape{n}_{key[0]}badges_{key[1]}pokemon"
    shutil.copy(os.path.join(ROM_DIR, f"{ROM_NAME}.ml{slot}"), os.path.join("sauvegardes", f"{label}.ml{slot}"))
    progress["slot"], progress["record"] = slot, key
    progress["history"].append({"etape": n, "badges": key[0], "equipe": key[1], "slot": slot,
                                "fenetre": i + 1, "total_steps": total_steps, "heure": time.strftime("%H:%M:%S")})
    json.dump(progress, open(PROGRESS, "w"), indent=2)
    for e in envs:
        e.set_start(slot)
    print(f"\n*** ETAPE {n} : {key[0]} badge(s), {key[1]} Pokemon (fenetre {i + 1}) -> sauvegarde {slot}, "
          f"tout le monde repartira de la ***\n")


calm_outside = [0] * N   # par fenetre : nombre d'actions d'affilee dehors, sans dialogue ni OPTIONS
previous_pos = [None] * N


def check_exit(i, info):
    """La premiere fenetre bien sortie de la maison sauvegarde (UNE seule fois), tout le monde repartira de la."""
    if progress.get("sortie"):
        return
    e = envs[i]
    calm = info["pos"][0] == EXIT_MAP and not info["options"] and not e.textbox_visible(e.frames[-1])
    calm_outside[i] = calm_outside[i] + 1 if calm else 0
    just_walked = previous_pos[i] is not None and info["pos"] != previous_pos[i]  # donc aucun menu ouvert
    previous_pos[i] = info["pos"]
    if calm_outside[i] < EXIT_CALM_STEPS or not just_walked:
        return
    e.emu.save_state(EXIT_SLOT)
    time.sleep(1.0)  # laisse melonDS ecrire le fichier
    shutil.copy(os.path.join(ROM_DIR, f"{ROM_NAME}.ml{EXIT_SLOT}"),
                os.path.join("sauvegardes", f"sortie_maison.ml{EXIT_SLOT}"))
    progress["sortie"] = {"fenetre": i + 1, "position": list(info["pos"]), "total_steps": total_steps,
                          "date": now()}
    if progress["slot"] == EXIT_SLOT:  # (si une etape plus loin existe deja, on ne revient pas en arriere)
        for env in envs:
            env.set_start(EXIT_SLOT)
    json.dump(progress, open(PROGRESS, "w"), indent=2)
    print(f"\n*** SORTIE DE LA MAISON (fenetre {i + 1}, {info['pos']}) -> sauvegarde {EXIT_SLOT} : "
          f"tout le monde repartira de dehors ***\n")


HEALTH_EVERY = 32       # toutes les 32 actions, on verifie que chaque fenetre repond encore
frozen_checks = [0] * N  # par fenetre : verifications d'affilee ou elle etait figee


def make_env(hwnd):
    return PokemonEnv(hwnd, max_steps=1024, save_slot=progress["slot"], coach=coach,
                      read_text=not args.sans_dialogues, map_places=places, visits=visits)


def check_health():
    """melonDS peut se figer ('ne repond pas') : le jeu tourne mais la fenetre ne lit plus les touches.
    Figee 2 verifications de suite -> on la ferme, on rouvre un melonDS et on la remet dans l'entrainement."""
    for i, e in enumerate(envs):
        frozen_checks[i] = frozen_checks[i] + 1 if is_frozen(e.emu.hwnd) else 0
        if frozen_checks[i] < 2:
            continue
        print(f"\n[reparation] la fenetre {i + 1} ne repond plus : je la ferme et j'en rouvre une\n")
        kill_window_process(e.emu.hwnd)
        hwnd = launch_instance()
        if hwnd is None:
            print("[reparation] le nouveau melonDS n'est pas apparu, nouvel essai plus tard")
            continue
        envs[i] = make_env(hwnd)
        obs[i] = envs[i].reset()
        ep_reward[i], ep_start[i], frozen_checks[i] = 0.0, time.time(), 0
        calm_outside[i], previous_pos[i] = 0, None
        print(f"[reparation] fenetre {i + 1} remplacee, elle rejoue")


def play(i, action):
    """Une action dans la fenetre i ; si l'episode est fini, on recommence depuis la sauvegarde de depart."""
    obs, reward, done, info = envs[i].step(action)
    if done:
        obs = envs[i].reset()
    return obs, reward, done, info


def log_episode(i, reward, info, seconds):
    global episode
    episode += 1
    print(f"Episode {episode:4d} [fenetre {i + 1}] | recompense {reward:6.2f} | cases {info['tiles']:4d} "
          f"| cartes {info['maps']} | dialogues {info['texts']} (relus {info['repeats']}) "
          f"| equipe {info['party']} (niv. {info['levels']}) "
          f"| badges {info['badges']} | {seconds:.0f}s"
          + (f" | BLOQUEE apres {info['steps']} actions" if info["events"]["bloque"] else ""))
    target = coach.target if coach else ""
    reached = int(bool(target) and target in envs[i].coach_rewarded)
    points, events = info["points"], info["events"]
    append_csv(LOG, {
        "date": now(), "session": SESSION, "episode": episode, "total_steps": total_steps, "window": i + 1,
        "start_slot": envs[i].save_slot, "reward": round(reward, 2), "tiles": info["tiles"], "maps": info["maps"],
        "texts": info["texts"], "repeats": info["repeats"], "party": info["party"], "levels": info["levels"],
        "badges": info["badges"], "coach_target": target, "reached_target": reached, "seconds": round(seconds),
        "coach_quest": (coach.objective or "") if coach else "",
        "coach_step": (coach.text("etape") or "") if coach else "",
        # d'ou viennent les points
        **{f"pts_{k}": round(points.get(k, 0.0), 2) for k in
           ("cases", "cartes", "lieux", "dialogues", "histoire", "pnj_quete", "relus", "murs", "menu", "bloque", "options", "sauvegarde",
            "coach", "equipe", "niveaux", "badges", "ko")},
        # ce qui s'est passe
        "options_entries": events["options"], "save_prompts": events["sauvegarde"], "blackouts": events["ko"],
        "wall_bumps": events["murs"], "menu_presses": events["menu"], "stuck_end": events["bloque"], "steps": info["steps"],
        "places": " > ".join(info["places"]),
        "dialogues": " | ".join(l[:40] for l in info["lines"][:6]),
    })


dashboard = None
if args.voir or args.video:
    from viewer import Dashboard
    if not 1 <= args.fenetre <= N:
        raise SystemExit(f"--fenetre {args.fenetre} : il n'y a que {N} fenetre(s)")
    dashboard = Dashboard(N, focus=args.fenetre - 1, capture=args.capture, show=args.voir,
                          record_path=args.video)

obs = np.stack(list(pool.map(lambda e: e.reset(), envs)))
if progress["record"] is None:  # tout premier lancement : le record = l'etat de la sauvegarde de depart
    progress["record"] = milestone_key(envs[0].progress)
    json.dump(progress, open(PROGRESS, "w"), indent=2)
ep_reward, ep_start = np.zeros(N), np.full(N, time.time())
print(f"C'est parti : {N} fenetre(s), cerveau sur {device}. Ctrl+C pour arreter.")

try:
    while True:
        # 1) Chaque fenetre joue ROLLOUT actions ; on note tout ce qui se passe
        buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones")}
        t0 = time.time()
        for _ in range(ROLLOUT):
            watchdog()
            if (total_steps // N) % HEALTH_EVERY == 0:
                check_health()
            actions, logp, values, probs = agent.act(obs)   # un seul calcul GPU pour toutes les fenetres
            results = list(pool.map(play, range(N), actions))  # toutes les fenetres jouent en meme temps
            rewards = np.array([r[1] for r in results])
            dones = np.array([float(r[2]) for r in results])
            for k, v in zip(buf, (obs, actions, logp, values, rewards, dones)):
                buf[k].append(v)
            obs = np.stack([r[0] for r in results])
            ep_reward += rewards
            total_steps += N
            if coach:
                coach.maybe_ask()  # en arriere-plan, seulement s'il y a du nouveau
            if dashboard:
                dashboard.update(envs, probs, actions, values, ep_reward, [r[3] for r in results],
                                 episode, total_steps)
            for i, (_, _, done, info) in enumerate(results):
                if done:
                    log_episode(i, ep_reward[i], info, time.time() - ep_start[i])
                    ep_reward[i], ep_start[i] = 0.0, time.time()
                    calm_outside[i], previous_pos[i] = 0, None
                else:  # (si l'episode vient de finir, la fenetre a deja ete remise au depart)
                    check_milestone(i, info)
                    check_exit(i, info)
        speed = ROLLOUT * N / (time.time() - t0)

        # 2) On calcule les avantages (par fenetre) puis on met a jour le cerveau avec tout le lot
        adv, returns = agent.advantages(buf["rewards"], buf["values"], buf["dones"], agent.value(obs))
        flat = lambda x: np.asarray(x).reshape(ROLLOUT * N, *np.asarray(x).shape[2:])
        pl, vl, ent = agent.update(flat(buf["obs"]), flat(buf["actions"]), flat(buf["logp"]),
                                   flat(adv), flat(returns))
        print(f"  maj @ {total_steps} actions | {speed:.1f} actions/s | perte politique {pl:+.3f} "
              f"| perte valeur {vl:.3f} | entropie {ent:.2f} (max {math.log(len(ACTIONS)):.2f} = totalement au hasard)")
        agent.save(CHECKPOINT, {"total_steps": total_steps, "episode": episode})
        visits.save()

except KeyboardInterrupt:
    print("Arret demande, sauvegarde...")
    agent.save(CHECKPOINT, {"total_steps": total_steps, "episode": episode})
    visits.save()
finally:
    if dashboard:
        dashboard.close()
    for e in envs:
        e.close()
