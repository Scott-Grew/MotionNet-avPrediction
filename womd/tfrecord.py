"""Reads and writes TFRecord framing, with its CRC-32C checks, and parses each
record into a Scenario proto.
"""
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterator

# From the file format itself: the CRC-32C (Castagnoli) polynomial
# in bit-reversed form, and the constant TFRecord adds when it
# masks a checksum. Checksums are 32-bit, hence the modulus.
CASTAGNOLI_REVERSED_POLYNOMIAL = 0x82F63B78
CRC_MASK_DELTA = 0xA282EAD8
UINT32_MODULUS = 0x100000000


def _build_crc32c_table() -> tuple[int, ...]:
    """Builds the 256-entry CRC-32C lookup table from the Castagnoli polynomial,
    for the table-driven crc32c below.
    """
    table = []
    for byte_value in range(256):
        remainder = byte_value
        for _ in range(8):
            if remainder & 1:
                remainder = (remainder >> 1) ^ CASTAGNOLI_REVERSED_POLYNOMIAL
            else:
                remainder >>= 1
        table.append(remainder)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(payload: bytes) -> int:
    """Computes the CRC-32C checksum of payload bytes using the table above;
    this is the checksum TFRecord framing uses.
    """
    remainder = 0xFFFFFFFF
    for byte_value in payload:
        table_index = (remainder ^ byte_value) & 0xFF
        remainder = _CRC32C_TABLE[table_index] ^ (remainder >> 8)
    return remainder ^ 0xFFFFFFFF


def mask_crc(value: int) -> int:
    """Applies TFRecord's CRC masking: rotates right 15 bits and adds a magic
    delta, per the TFRecord/LevelDB framing spec.
    """
    rotated = ((value >> 15) | (value << 17)) % UINT32_MODULUS
    return (rotated + CRC_MASK_DELTA) % UINT32_MODULUS


class CorruptRecordError(Exception):
    """Raised when a TFRecord record's framing or checksum comes out truncated
    or wrong.
    """

    pass


def read_records(stream: BinaryIO,
                 verify_checksums: bool = True) -> Iterator[bytes]:
    """Reads length-prefixed TFRecord records from a binary stream and yields
    their raw payloads, checking CRCs when asked.
    """
    while True:
        length_bytes = stream.read(8)
        if not length_bytes:
            return
        if len(length_bytes) != 8:
            raise CorruptRecordError("truncated record length")
        length_checksum_bytes = stream.read(4)
        if len(length_checksum_bytes) != 4:
            raise CorruptRecordError("truncated length checksum")
        payload_length = struct.unpack("<Q", length_bytes)[0]
        payload = stream.read(payload_length)
        if len(payload) != payload_length:
            raise CorruptRecordError("truncated record payload")
        payload_checksum_bytes = stream.read(4)
        if len(payload_checksum_bytes) != 4:
            raise CorruptRecordError("truncated payload checksum")
        if verify_checksums:
            expected_length_checksum = struct.unpack("<I",
                                                     length_checksum_bytes)[0]
            if mask_crc(crc32c(length_bytes)) != expected_length_checksum:
                raise CorruptRecordError("record length checksum mismatch")
            expected_payload_checksum = struct.unpack("<I",
                                                      payload_checksum_bytes)[0]
            if mask_crc(crc32c(payload)) != expected_payload_checksum:
                raise CorruptRecordError("record payload checksum mismatch")
        yield payload


def write_record(stream: BinaryIO, payload: bytes) -> None:
    """Writes one payload to a stream in TFRecord framing: length, length
    checksum, payload, then payload checksum.
    """
    length_bytes = struct.pack("<Q", len(payload))
    stream.write(length_bytes)
    stream.write(struct.pack("<I", mask_crc(crc32c(length_bytes))))
    stream.write(payload)
    stream.write(struct.pack("<I", mask_crc(crc32c(payload))))


def read_scenarios(path: Path | str,
                   scenario_message_type: type,
                   verify_checksums: bool = True) -> Iterator[Any]:
    """Opens a TFRecord shard and yields each record parsed into
    scenario_message_type.
    """
    with open(path, "rb") as stream:
        for payload in read_records(stream, verify_checksums):
            scenario = scenario_message_type()
            scenario.ParseFromString(payload)
            yield scenario
