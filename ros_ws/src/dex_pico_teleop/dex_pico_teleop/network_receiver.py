"""Threaded TCP/UDP receiver for Pico JSON packets."""

from __future__ import annotations

import json
import queue
import socket
import threading
from collections.abc import Callable

from dex_pico_teleop.xr_packet import PicoPacket


LogFn = Callable[[str], None]
XROBOT_CLIENT_HEAD = 0x3F
XROBOT_SERVER_HEAD = 0xCF
XROBOT_END = 0xA5
XROBOT_TRACKING_FUNCTION_CMD = 0x6D
XROBOT_MIN_PACKET_SIZE = 15


class NetworkReceiver:
    """Receive PicoPacket objects without blocking the ROS executor."""

    def __init__(
        self,
        transport: str = "udp",
        host: str = "0.0.0.0",
        port: int = 63901,
        max_queue: int = 8,
        log_info: LogFn | None = None,
        log_warn: LogFn | None = None,
    ) -> None:
        self.transport = transport.lower()
        self.host = host
        self.port = int(port)
        self._queue: queue.Queue[PicoPacket] = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._log_info = log_info or (lambda _msg: None)
        self._log_warn = log_warn or (lambda _msg: None)

    def start(self) -> None:
        if self._thread is not None:
            return
        target = self._run_udp if self.transport == "udp" else self._run_tcp
        if self.transport not in {"udp", "tcp"}:
            raise ValueError("transport must be 'udp' or 'tcp'")
        self._thread = threading.Thread(target=target, name="pico_json_receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self) -> PicoPacket | None:
        latest: PicoPacket | None = None
        for packet in self.get_available():
            latest = packet
        return latest

    def get_available(self) -> list[PicoPacket]:
        packets: list[PicoPacket] = []
        while True:
            try:
                packets.append(self._queue.get_nowait())
            except queue.Empty:
                return packets

    def _run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket = sock
        sock.settimeout(0.2)
        sock.bind((self.host, self.port))
        self._log_info(f"listening for Pico UDP JSON on {self.host}:{self.port}")
        while not self._stop_event.is_set():
            try:
                payload, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self._parse_and_push(payload)

    def _run_tcp(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(0.2)
        server.bind((self.host, self.port))
        server.listen(1)
        self._log_info(
            f"listening for Pico TCP JSON/XRoboToolkit packets on {self.host}:{self.port}"
        )
        while not self._stop_event.is_set():
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._log_info(f"Pico TCP client connected from {addr}")
            with client:
                client.settimeout(0.2)
                buffer = b""
                while not self._stop_event.is_set():
                    try:
                        chunk = client.recv(65535)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buffer += chunk
                    buffer = self._drain_tcp_buffer(buffer)

    def _drain_tcp_buffer(self, buffer: bytes) -> bytes:
        while buffer:
            stripped = buffer.lstrip()
            if len(stripped) != len(buffer):
                buffer = stripped
            if not buffer:
                return buffer

            if buffer[0] in (ord("{"), ord("[")):
                if b"\n" not in buffer:
                    return buffer
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    self._parse_and_push(line)
                continue

            if buffer[0] in (XROBOT_CLIENT_HEAD, XROBOT_SERVER_HEAD):
                if len(buffer) < XROBOT_MIN_PACKET_SIZE:
                    return buffer
                payload_len = int.from_bytes(buffer[2:6], byteorder="little", signed=False)
                packet_len = XROBOT_MIN_PACKET_SIZE + payload_len
                if packet_len > 1_000_000:
                    self._log_warn(f"discarding oversized XRoboToolkit packet length {payload_len}")
                    buffer = buffer[1:]
                    continue
                if len(buffer) < packet_len:
                    return buffer
                packet, buffer = buffer[:packet_len], buffer[packet_len:]
                if packet[-1] != XROBOT_END:
                    self._log_warn("discarding XRoboToolkit packet with invalid terminator")
                    continue
                self._parse_and_push_xrobotoolkit(packet[1], packet[6 : 6 + payload_len])
                continue

            self._log_warn(f"discarding unknown TCP byte 0x{buffer[0]:02x}")
            buffer = buffer[1:]
        return buffer

    def _parse_and_push(self, payload: bytes) -> None:
        try:
            packet = PicoPacket.from_json_bytes(payload)
        except Exception as exc:  # noqa: BLE001 - this is a network boundary
            self._log_warn(f"ignoring malformed Pico packet: {exc}")
            return
        self._push_packet(packet)

    def _parse_and_push_xrobotoolkit(self, command: int, payload: bytes) -> None:
        if command != XROBOT_TRACKING_FUNCTION_CMD:
            return
        try:
            message = json.loads(payload.decode("utf-8"))
            if str(message.get("functionName", "")).lower() != "tracking":
                return
            value = message.get("value", {})
            tracking = json.loads(value) if isinstance(value, str) else value
            packet = PicoPacket.from_xrobotoolkit_tracking(tracking)
        except Exception as exc:  # noqa: BLE001 - this is a network boundary
            self._log_warn(f"ignoring malformed XRoboToolkit tracking packet: {exc}")
            return
        self._push_packet(packet)

    def _push_packet(self, packet: PicoPacket) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(packet)
