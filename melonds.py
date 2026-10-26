"""Controle de melonDS : trouver les fenetres, capturer l'ecran, appuyer sur les touches.

Tout fonctionne en arriere-plan (PostMessage + PrintWindow) : plusieurs melonDS peuvent jouer
en meme temps, caches derriere d'autres fenetres, pendant que tu utilises ton PC.
"""
import ctypes
import os
import subprocess
import time
from ctypes import wintypes

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32")
kernel32 = ctypes.WinDLL("kernel32")
ctypes.windll.shcore.SetProcessDpiAwareness(2)  # coordonnees en vrais pixels
# Sans ca, Windows arrondit chaque time.sleep au multiple de 15,6 ms superieur (sleep(0.001) = 15,5 ms,
# sleep(0.022) = 31 ms) : toutes les attentes du jeu etaient plus longues que prevu.
ctypes.WinDLL("winmm").timeBeginPeriod(1)

# Touches configurees dans melonDS (codes virtuels Windows -> independant AZERTY/QWERTY)
VK = {
    "A": 0x41, "B": 0x42, "X": 0x58, "Y": 0x59, "L": 0x4C, "R": 0x52,
    "START": 0x53,   # touche S
    "SELECT": 0x0D,  # Entree
    "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
    "SHIFT": 0x10,
    **{f"F{i}": 0x6F + i for i in range(1, 9)},
    # Raccourcis melonDS (pave numerique)
    "PAUSE": 0x61, "RESET": 0x62, "FRAME_STEP": 0x63,
    "FAST_FORWARD": 0x64, "TOGGLE_FAST_FORWARD": 0x65, "SLOW_MO": 0x66,
}
EXTENDED = {VK["UP"], VK["DOWN"], VK["LEFT"], VK["RIGHT"]}

WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
WM_ACTIVATE, WA_ACTIVE, WM_SETFOCUS = 0x0006, 1, 0x0007
VK_SHIFT, VK_LSHIFT = 0x10, 0xA0
SMTO_ABORTIFHUNG = 0x0002
PW_CLIENTONLY, PW_RENDERFULLCONTENT = 0x1, 0x2  # sans FULLCONTENT : 5 ms au lieu de 19, meme image


kernel32_ = ctypes.WinDLL("kernel32", use_last_error=True)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def process_name(hwnd):
    """Nom de l'executable proprietaire de la fenetre (ex. 'melonDS.exe')."""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32_.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    buf, size = ctypes.create_unicode_buffer(512), wintypes.DWORD(512)
    ok = kernel32_.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
    kernel32_.CloseHandle(handle)
    return os.path.basename(buf.value) if ok else ""


def find_windows():
    """Toutes les fenetres de jeu melonDS ("[60/60] melonDS 1.1"), pas les boites de dialogue.
    Quand melonDS se fige, Windows affiche une fenetre FANTOME ("ne repond pas") au meme titre,
    qui appartient a un processus systeme : on ne garde que les fenetres de melonDS.exe."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            # "Input and hotkeys - melonDS" ou "Emu settings - melonDS" sont des dialogues
            if ("melonDS" in buf.value and " - melonDS" not in buf.value
                    and process_name(hwnd).lower() == "melonds.exe"):
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found


def find_window():
    found = find_windows()
    return found[0] if found else (None, None)


MELONDS_EXE = r"C:\Users\enzop\Desktop\melonDS.exe"
ROM_PATH = (r"C:\Users\enzop\Desktop\Pokemon - Diamond Version (Europe) (Rev 5)"
            r"\Pokemon - Diamond Version (Europe) (Rev 5).nds")


def launch_instance(timeout=30):
    """Ouvre un nouveau melonDS avec la ROM et renvoie sa fenetre (ou None si elle n'apparait pas)."""
    before = {h for h, _ in find_windows()}
    subprocess.Popen([MELONDS_EXE, ROM_PATH], cwd=os.path.dirname(MELONDS_EXE))
    t = time.time()
    while time.time() - t < timeout:
        time.sleep(1)
        new = [h for h, title in find_windows() if h not in before and title.startswith("[")]
        if new:
            time.sleep(3)  # laisse la ROM demarrer
            return new[0]
    return None


def is_frozen(hwnd):
    """La fenetre ne repond plus aux messages (melonDS 'ne repond pas') : elle ne recevra plus les touches."""
    if not user32.IsWindow(hwnd):
        return True
    result = ctypes.c_size_t()
    return bool(user32.IsHungAppWindow(hwnd)) or not user32.SendMessageTimeoutW(
        hwnd, 0, 0, 0, SMTO_ABORTIFHUNG, 1000, ctypes.byref(result))


def kill_window_process(hwnd):
    """Ferme le melonDS de cette fenetre. Refuse tout autre programme (securite)."""
    if process_name(hwnd).lower() != "melonds.exe":
        print(f"[reparation] la fenetre n'appartient pas a melonDS ({process_name(hwnd) or '?'}) : je ne ferme rien")
        return False
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    subprocess.run(["taskkill", "/F", "/PID", str(pid.value)], capture_output=True)
    return True


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class MelonDS:
    def __init__(self, hwnd=None, hold=0.08):
        if hwnd is None:
            hwnd, _ = find_window()
        if not hwnd:
            raise RuntimeError("Fenetre melonDS introuvable : lance melonDS et la ROM d'abord.")
        self.hwnd = hwnd
        self.hold = hold
        self._gdi = None  # (taille, DC, bitmap, tampon) gardes d'une capture a l'autre

    def title(self):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(self.hwnd, buf, 256)
        return buf.value

    # ---------- yeux ----------
    def screenshot(self):
        """Image RGB (numpy) de la zone client, meme si la fenetre est cachee."""
        r = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        w, h = r.right, r.bottom
        if self._gdi is None or self._gdi[0] != (w, h):  # premiere capture, ou fenetre redimensionnee
            self._free_gdi()
            hdc = user32.GetDC(self.hwnd)
            mdc = gdi32.CreateCompatibleDC(hdc)
            bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
            user32.ReleaseDC(self.hwnd, hdc)
            gdi32.SelectObject(mdc, bmp)
            self._gdi = ((w, h), mdc, bmp, ctypes.create_string_buffer(w * h * 4))
        _, mdc, bmp, buf = self._gdi
        bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        for flags in (PW_CLIENTONLY, PW_CLIENTONLY | PW_RENDERFULLCONTENT):
            user32.PrintWindow(self.hwnd, mdc, flags)
            gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
            img = np.frombuffer(buf.raw, np.uint8).reshape(h, w, 4)
            if img[::16, ::16, :3].max() > 0:  # image noire (fenetre reduite...) : mode lent, plus sur
                break
        return img[:, :, 2::-1]  # BGRA -> RGB

    def _free_gdi(self):
        if self._gdi:
            gdi32.DeleteObject(self._gdi[2])
            gdi32.DeleteDC(self._gdi[1])
            self._gdi = None

    # ---------- mains ----------
    def key(self, button, up):
        vk = VK[button]
        lp = 1 | (user32.MapVirtualKeyW(vk, 0) << 16)
        if vk in EXTENDED:
            lp |= 1 << 24
        if up:
            lp |= 0xC0000000
        user32.PostMessageW(self.hwnd, WM_KEYUP if up else WM_KEYDOWN, vk, lp)

    def press(self, button, hold=None):
        self.key(button, up=False)
        time.sleep(hold or self.hold)
        self.key(button, up=True)

    def toggle_fast_forward(self):
        self.press("TOGGLE_FAST_FORWARD")

    def _send(self, msg, wparam, lparam):
        """SendMessage avec une limite de 1 s : si melonDS ne repond pas, on n'attend pas pour toujours
        (un SendMessage simple pouvait figer tout l'entrainement)."""
        result = ctypes.c_size_t()
        user32.SendMessageTimeoutW(self.hwnd, msg, wparam, lparam, SMTO_ABORTIFHUNG, 1000, ctypes.byref(result))

    def _activate(self):
        # F1..F8 sont des raccourcis de menu Qt : ils n'agissent que si la fenetre se croit active.
        # On le lui fait croire (sans lui donner vraiment le premier plan) juste avant d'appuyer.
        self._send(WM_ACTIVATE, WA_ACTIVE, 0)
        self._send(WM_SETFOCUS, 0, 0)

    def load_state(self, slot=1):  # melonDS : F1..F8 (fichier .mlN a cote de la ROM)
        self._activate()
        self.press(f"F{slot}")

    def save_state(self, slot):  # melonDS : Shift+F1..F8
        """Qt lit l'etat de Maj dans l'etat clavier du fil de melonDS, pas dans le message.
        On s'attache a ce fil pour partager son etat clavier, on y 'enfonce' Maj, on appuie, on remet."""
        target = user32.GetWindowThreadProcessId(self.hwnd, None)
        me = kernel32.GetCurrentThreadId()
        user32.AttachThreadInput(me, target, True)
        try:
            keys = (ctypes.c_ubyte * 256)()
            user32.GetKeyboardState(keys)
            old = keys[VK_SHIFT], keys[VK_LSHIFT]
            keys[VK_SHIFT] = keys[VK_LSHIFT] = 0x80
            user32.SetKeyboardState(keys)
            self._activate()
            self.press(f"F{slot}")
            time.sleep(0.3)
            keys[VK_SHIFT], keys[VK_LSHIFT] = old
            user32.SetKeyboardState(keys)
        finally:
            user32.AttachThreadInput(me, target, False)
