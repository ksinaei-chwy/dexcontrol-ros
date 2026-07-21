from types import SimpleNamespace

import dex_vega_lerobot_recorder.input_controller as input_controller
from dex_vega_lerobot_recorder.input_controller import KeyboardInputController


def make_controller(*, on_key=lambda _key: None, info=None, warnings=None):
    info = [] if info is None else info
    warnings = [] if warnings is None else warnings
    return KeyboardInputController(
        backend="terminal",
        device_path="",
        debounce_seconds=0.0,
        on_key=on_key,
        log_info=info.append,
        log_warn=warnings.append,
    )


def test_terminal_backend_prefers_interactive_standard_input(monkeypatch):
    stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 17)
    monkeypatch.setattr(input_controller.sys, "stdin", stdin)

    def unexpected_open(_path, _flags):
        raise AssertionError("/dev/tty should not be opened for TTY stdin")

    monkeypatch.setattr(input_controller.os, "open", unexpected_open)
    assert make_controller()._acquire_terminal_fd() == (17, False)


def test_terminal_backend_reads_controlling_tty_when_launch_stdin_is_pipe(
    monkeypatch,
):
    keys = []
    info = []
    warnings = []
    closed = []
    restored = []
    controller = make_controller(
        on_key=keys.append,
        info=info,
        warnings=warnings,
    )
    stdin = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(input_controller.sys, "stdin", stdin)
    monkeypatch.setattr(input_controller.os, "open", lambda path, flags: 42)
    monkeypatch.setattr(input_controller.os, "isatty", lambda fd: fd == 42)
    monkeypatch.setattr(input_controller.os, "close", closed.append)
    monkeypatch.setattr(input_controller.termios, "tcgetattr", lambda fd: [fd])
    monkeypatch.setattr(input_controller.tty, "setcbreak", lambda fd: None)
    monkeypatch.setattr(
        input_controller.termios,
        "tcsetattr",
        lambda fd, when, settings: restored.append((fd, when, settings)),
    )

    def ready_once(readers, _writers, _errors, _timeout):
        return readers, [], []

    monkeypatch.setattr(input_controller.select, "select", ready_once)

    def read_key(fd, size):
        assert (fd, size) == (42, 1)
        controller._stop.set()
        return b"a"

    monkeypatch.setattr(input_controller.os, "read", read_key)
    controller._run_terminal()

    assert keys == ["a"]
    assert warnings == []
    assert any("using controlling terminal /dev/tty" in line for line in info)
    assert "terminal keyboard input active" in info
    assert closed == [42]
    assert restored == [(42, input_controller.termios.TCSADRAIN, [42])]


def test_terminal_backend_warns_when_no_controlling_terminal(monkeypatch):
    warnings = []
    stdin = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(input_controller.sys, "stdin", stdin)

    def unavailable(_path, _flags):
        raise OSError("no controlling terminal")

    monkeypatch.setattr(input_controller.os, "open", unavailable)
    fd, owns_fd = make_controller(warnings=warnings)._acquire_terminal_fd()
    assert fd is None
    assert owns_fd is False
    assert "no controlling TTY" in warnings[0]
