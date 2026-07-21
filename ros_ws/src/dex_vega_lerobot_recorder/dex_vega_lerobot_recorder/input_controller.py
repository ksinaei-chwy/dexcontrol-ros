"""Headless Linux input-event and terminal keyboard control backends."""

from __future__ import annotations

import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Callable


_INPUT_EVENT = struct.Struct("llHHI")
_EV_KEY = 0x01
_LETTER_KEY_CODES = dict(
    zip(
        (*range(16, 26), *range(30, 39), *range(44, 51)),
        "qwertyuiopasdfghjklzxcvbnm",
    )
)


class KeyDebouncer:
    def __init__(self, debounce_seconds: float, monotonic: Callable[[], float] = time.monotonic):
        self.debounce_seconds = float(debounce_seconds)
        self._monotonic = monotonic
        self._last_press: dict[str, float] = {}

    def accept(self, key: str) -> bool:
        now = self._monotonic()
        previous = self._last_press.get(key, float("-inf"))
        if now - previous < self.debounce_seconds:
            return False
        self._last_press[key] = now
        return True


class KeyboardInputController:
    """Read one key-down event per press without requiring X11."""

    def __init__(
        self,
        *,
        backend: str,
        device_path: str,
        debounce_seconds: float,
        on_key: Callable[[str], None],
        log_info: Callable[[str], None] | None = None,
        log_warn: Callable[[str], None] | None = None,
    ) -> None:
        self.backend = backend
        self.device_path = device_path
        self.on_key = on_key
        self._debouncer = KeyDebouncer(debounce_seconds)
        self._log_info = log_info or (lambda _message: None)
        self._log_warn = log_warn or (lambda _message: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.backend == "disabled":
            self._log_info("keyboard input disabled; ROS services remain available")
            return
        target = self._run_terminal if self.backend == "terminal" else self._run_event_device
        self._thread = threading.Thread(
            target=target, name=f"recorder-input-{self.backend}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _dispatch(self, key: str) -> None:
        key = key.lower()
        if self._debouncer.accept(key):
            try:
                self.on_key(key)
            except Exception as exc:  # noqa: BLE001 - input thread boundary
                self._log_warn(f"key '{key}' command failed: {exc}")

    def _acquire_terminal_fd(self) -> tuple[int | None, bool]:
        """Return a terminal input fd and whether this controller owns it."""
        if sys.stdin.isatty():
            return sys.stdin.fileno(), False

        flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open("/dev/tty", flags)
        except OSError as exc:
            self._log_warn(
                "terminal keyboard backend has no controlling TTY; use ROS "
                f"services or linux_input_event for physical pedals: {exc}"
            )
            return None, False
        if not os.isatty(fd):
            os.close(fd)
            self._log_warn(
                "terminal keyboard backend opened /dev/tty but it is not a TTY; "
                "use ROS services or linux_input_event for physical pedals"
            )
            return None, False
        self._log_info(
            "standard input is not a TTY; using controlling terminal /dev/tty"
        )
        return fd, True

    def _run_terminal(self) -> None:
        fd, owns_fd = self._acquire_terminal_fd()
        if fd is None:
            return
        previous: list | None = None
        try:
            previous = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._log_info("terminal keyboard input active")
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    try:
                        raw = os.read(fd, 1)
                    except BlockingIOError:
                        continue
                    key = raw.decode("utf-8", errors="ignore")
                    if key:
                        self._dispatch(key)
        except (OSError, termios.error) as exc:
            self._log_warn(f"terminal keyboard backend failed: {exc}")
        finally:
            if previous is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, previous)
                except (OSError, termios.error) as exc:
                    self._log_warn(f"failed to restore terminal settings: {exc}")
            if owns_fd:
                os.close(fd)

    def _run_event_device(self) -> None:
        path = Path(self.device_path)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            self._log_warn(f"cannot open pedal input device {path}: {exc}")
            return
        self._log_info(f"Linux input-event pedal backend active on {path}")
        buffer = bytearray()
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, _INPUT_EVENT.size * 16)
                except BlockingIOError:
                    continue
                buffer.extend(chunk)
                while len(buffer) >= _INPUT_EVENT.size:
                    raw = bytes(buffer[: _INPUT_EVENT.size])
                    del buffer[: _INPUT_EVENT.size]
                    _sec, _usec, event_type, code, value = _INPUT_EVENT.unpack(raw)
                    if event_type == _EV_KEY and value == 1:
                        key = _LETTER_KEY_CODES.get(code)
                        if key is not None:
                            self._dispatch(key)
        finally:
            os.close(fd)
