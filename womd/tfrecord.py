import struct

CASTAGNOLI_REVERSED_POLYNOMIAL = 0x82F63B78
CRC_MASK_DELTA = 0xA282EAD8
UINT32_MODULUS = 0x100000000


def _build_crc32c_table():
    table = []
    for byte_value in range(256):
        remainder = byte_value
        for _ in range(8):
            if remainder & 1:
                remainder = (
                    remainder >> 1
                ) ^ CASTAGNOLI_REVERSED_POLYNOMIAL
            else:
                remainder >>= 1
        table.append(remainder)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(payload):
    remainder = 0xFFFFFFFF
    for byte_value in payload:
        remainder = _CRC32C_TABLE[(remainder ^ byte_value) & 0xFF] ^ (
            remainder >> 8
        )
    return remainder ^ 0xFFFFFFFF


def mask_crc(value):
    rotated = ((value >> 15) | (value << 17)) % UINT32_MODULUS
    return (rotated + CRC_MASK_DELTA) % UINT32_MODULUS


class CorruptRecordError(Exception):
    pass


def read_records(stream, verify_checksums=True):
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
            expected_length_checksum = struct.unpack(
                "<I", length_checksum_bytes
            )[0]
            if (
                mask_crc(crc32c(length_bytes))
                != expected_length_checksum
            ):
                raise CorruptRecordError(
                    "record length checksum mismatch"
                )
            expected_payload_checksum = struct.unpack(
                "<I", payload_checksum_bytes
            )[0]
            if mask_crc(crc32c(payload)) != expected_payload_checksum:
                raise CorruptRecordError(
                    "record payload checksum mismatch"
                )
        yield payload


def write_record(stream, payload):
    length_bytes = struct.pack("<Q", len(payload))
    stream.write(length_bytes)
    stream.write(struct.pack("<I", mask_crc(crc32c(length_bytes))))
    stream.write(payload)
    stream.write(struct.pack("<I", mask_crc(crc32c(payload))))


def read_scenarios(
    path, scenario_message_type, verify_checksums=True
):
    with open(path, "rb") as stream:
        for payload in read_records(stream, verify_checksums):
            scenario = scenario_message_type()
            scenario.ParseFromString(payload)
            yield scenario
