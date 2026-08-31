"""Loss-aware DCA1000 UDP packet parsing and frame assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dca1000Packet:
    sequence: int
    byte_count: int
    payload: bytes


@dataclass(frozen=True)
class AssembledAdcFrame:
    payload: bytes
    valid_bytes: np.ndarray
    packets: int
    dropped_packets: int
    stream_start_byte: int
    stream_end_byte: int

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid_bytes)) if self.valid_bytes.size else 0.0

    @property
    def invalid_byte_ranges(self) -> list[tuple[int, int]]:
        """Half-open invalid byte ranges, compact enough for JSON manifests."""
        invalid = np.flatnonzero(~self.valid_bytes)
        if len(invalid) == 0:
            return []
        ranges: list[tuple[int, int]] = []
        start = previous = int(invalid[0])
        for value in invalid[1:]:
            current = int(value)
            if current != previous + 1:
                ranges.append((start, previous + 1))
                start = current
            previous = current
        ranges.append((start, previous + 1))
        return ranges


def parse_dca1000_packet(packet: bytes) -> Dca1000Packet:
    if len(packet) <= 10:
        raise ValueError(f"DCA1000 packet too short: {len(packet)}")
    return Dca1000Packet(
        sequence=int.from_bytes(packet[:4], "little", signed=False),
        byte_count=int.from_bytes(packet[4:10], "little", signed=False),
        payload=bytes(packet[10:]),
    )


class Dca1000FrameAssembler:
    """Assemble fixed-size frames without shifting data after UDP gaps.

    The six-byte DCA byte counter anchors every payload in the continuous ADC
    stream. Missing regions are zero-filled and marked invalid. Duplicated or
    overlapping payload bytes are discarded rather than appended twice.
    """

    def __init__(self, frame_bytes: int) -> None:
        if frame_bytes <= 0:
            raise ValueError("frame_bytes must be positive")
        self.frame_bytes = int(frame_bytes)
        self._data = bytearray()
        self._valid = bytearray()
        self._stream_offset: int | None = None
        self._buffer_start_offset: int | None = None
        self._last_sequence: int | None = None
        self._frame_packets = 0
        self._frame_dropped = 0
        self.total_packets = 0
        self.total_dropped = 0
        self.duplicate_bytes = 0

    def push(self, raw_packet: bytes) -> list[AssembledAdcFrame]:
        packet = parse_dca1000_packet(raw_packet)
        self.total_packets += 1
        self._frame_packets += 1
        if self._last_sequence is not None:
            gap_packets = (packet.sequence - self._last_sequence - 1) & 0xFFFFFFFF
            if 0 < gap_packets < 1_000_000:
                self.total_dropped += gap_packets
                self._frame_dropped += gap_packets
        self._last_sequence = packet.sequence

        if self._stream_offset is None:
            # Preserve the hardware frame boundary even when capture starts
            # after one or more packets have already been missed.
            self._stream_offset = packet.byte_count - (packet.byte_count % self.frame_bytes)
            self._buffer_start_offset = self._stream_offset

        expected = int(self._stream_offset)
        start = int(packet.byte_count)
        payload = packet.payload
        if start > expected:
            missing = start - expected
            self._data.extend(b"\x00" * missing)
            self._valid.extend(b"\x00" * missing)
            expected = start
        elif start < expected:
            overlap = expected - start
            if overlap >= len(payload):
                self.duplicate_bytes += len(payload)
                return self._pop_frames()
            self.duplicate_bytes += overlap
            payload = payload[overlap:]

        self._data.extend(payload)
        self._valid.extend(b"\x01" * len(payload))
        self._stream_offset = expected + len(payload)
        return self._pop_frames()

    def _pop_frames(self) -> list[AssembledAdcFrame]:
        frames: list[AssembledAdcFrame] = []
        while len(self._data) >= self.frame_bytes:
            if self._buffer_start_offset is None:
                raise RuntimeError("DCA assembler buffer offset is not initialized")
            frame_start = int(self._buffer_start_offset)
            payload = bytes(self._data[:self.frame_bytes])
            valid = np.frombuffer(bytes(self._valid[:self.frame_bytes]), dtype=np.uint8).astype(bool)
            del self._data[:self.frame_bytes]
            del self._valid[:self.frame_bytes]
            frames.append(AssembledAdcFrame(
                payload=payload,
                valid_bytes=valid,
                packets=self._frame_packets,
                dropped_packets=self._frame_dropped,
                stream_start_byte=frame_start,
                stream_end_byte=frame_start + self.frame_bytes,
            ))
            self._buffer_start_offset += self.frame_bytes
            self._frame_packets = 0
            self._frame_dropped = 0
        return frames

    @property
    def buffered_bytes(self) -> int:
        return len(self._data)
