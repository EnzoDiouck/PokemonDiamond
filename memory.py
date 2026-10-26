"""Lecture de la RAM de la DS emulee, directement dans la memoire du processus melonDS."""
import ctypes
import struct
from ctypes import wintypes

import numpy as np

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)

PROCESS_VM_READ, PROCESS_QUERY_INFORMATION = 0x0010, 0x0400
MEM_COMMIT = 0x1000
READABLE = {0x02, 0x04, 0x08, 0x20, 0x40, 0x80}  # PAGE_READONLY, READWRITE, ... EXECUTE_*

RAM_START = 0x02000000           # adresse de la RAM principale vue par la DS
RAM_SIZE = 0x400000              # 4 Mo
HEADER_OFFSET = 0x3FFE00         # copie de l'en-tete de la ROM en 0x027FFE00
HEADER_SIG = b"POKEMON D\x00\x00\x00ADAE"

kernel32.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
kernel32.VirtualQueryEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
kernel32.VirtualQueryEx.restype = ctypes.c_size_t


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD),
                ("RegionSize", ctypes.c_size_t), ("State", wintypes.DWORD),
                ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]


class DSMemory:
    def __init__(self, hwnd):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        self.handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid.value)
        if not self.handle:
            raise OSError(f"OpenProcess a echoue ({ctypes.get_last_error()})")
        self.base = self._find_ram()

    def _raw(self, addr, size):
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t()
        if not kernel32.ReadProcessMemory(self.handle, ctypes.c_void_p(addr), buf, size, ctypes.byref(n)):
            return None
        return buf.raw[:n.value]

    def _regions(self):
        addr, mbi = 0, MEMORY_BASIC_INFORMATION()
        while kernel32.VirtualQueryEx(self.handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            if mbi.State == MEM_COMMIT and mbi.Protect in READABLE:
                yield mbi.BaseAddress, mbi.RegionSize
            addr = (mbi.BaseAddress or 0) + mbi.RegionSize

    def find_header_copies(self):
        """Toutes les adresses (cote Windows) ou l'en-tete du jeu apparait dans melonDS."""
        hits = []
        for start, size in self._regions():
            if size > 0x40000000:
                continue
            data = self._raw(start, size)
            if not data:
                continue
            pos = data.find(HEADER_SIG)
            while pos != -1:
                hits.append((start + pos, start, size))
                pos = data.find(HEADER_SIG, pos + 1)
        return hits

    def _find_ram(self):
        """L'en-tete recopie en RAM DS est a 0x027FFE00 : son adresse - 0x3FFE00 = debut de la RAM.
        Le fichier ROM charge contient aussi l'en-tete (a son offset 0) : on l'ecarte car
        la zone de 4 Mo qui le precederait n'appartient pas a la meme allocation."""
        for addr, start, size in self.find_header_copies():
            base = addr - HEADER_OFFSET
            # melonDS alloue la RAM DS alignee sur une page (multiple de 0x1000)
            if base % 0x1000 == 0 and base >= start and base + RAM_SIZE <= start + size:
                return base
        raise RuntimeError("RAM DS introuvable dans melonDS (le jeu est-il lance ?)")

    # --- lecture avec les adresses de la DS (0x02xxxxxx) ---
    def read(self, ds_addr, size):
        # La RAM de 4 Mo est en miroir de 0x02000000 a 0x02FFFFFF
        return self._raw(self.base + ((ds_addr - RAM_START) & (RAM_SIZE - 1)), size)

    def u8(self, a):  return self.read(a, 1)[0]
    def u16(self, a): return struct.unpack("<H", self.read(a, 2))[0]
    def u32(self, a): return struct.unpack("<I", self.read(a, 4))[0]

    def snapshot(self):
        """Toute la RAM (4 Mo) sous forme de tableau numpy."""
        return np.frombuffer(self._raw(self.base, RAM_SIZE), dtype=np.uint8)
