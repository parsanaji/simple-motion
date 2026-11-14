"""Offline tests for Govee packet building. Run: uv run --with bleak --with pytest pytest -q"""

from functools import reduce
from operator import xor

import pytest

import govee_h6147 as g


@pytest.mark.parametrize(
    "packet, expected",
    [
        # Well-known reference packets from the Govee BLE community docs.
        (g.power_packet(True), "3301010000000000000000000000000000000033"),
        (g.power_packet(False), "3301000000000000000000000000000000000032"),
        (g.keepalive_packet(), "aa010000000000000000000000000000000000ab"),
        (g.VARIANTS_BY_NAME["rgb-02"].packet(g.RED), "330502ff000000000000000000000000000000cb"),
    ],
)
def test_reference_packets(packet, expected):
    assert packet.hex() == expected


@pytest.mark.parametrize("variant", g.VARIANTS, ids=lambda v: v.name)
def test_every_variant_is_valid_20_byte_packet(variant):
    packet = variant.packet(g.RED)
    assert len(packet) == 20
    assert packet[0] == 0x33
    assert packet[1] == g.CMD_COLOR
    assert packet[-1] == reduce(xor, packet[:-1])
    rgb_at = 2 + len(variant.mode)  # RGB sits right after the mode bytes
    assert tuple(packet[rgb_at : rgb_at + 3]) == g.RED


def test_rgbic_1501_layout():
    packet = g.VARIANTS_BY_NAME["rgbic-1501"].packet(g.RED)
    assert list(packet[:14]) == [
        0x33, 0x05, 0x15, 0x01, 0xFF, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x7F,
    ]
    assert set(packet[14:19]) == {0}


def test_checksum_xors_all_19_bytes():
    packet = g.build_packet(0x33, [0x05, 0x02, 0x12, 0x34, 0x56])
    assert packet[-1] == 0x33 ^ 0x05 ^ 0x02 ^ 0x12 ^ 0x34 ^ 0x56


def test_max_payload_ok_and_overflow_rejected():
    assert len(g.build_packet(0x33, [1] * 18)) == 20
    with pytest.raises(ValueError):
        g.build_packet(0x33, [1] * 19)


def test_name_matching():
    class Dev:
        def __init__(self, name):
            self.name = name

    class Adv:
        def __init__(self, local_name):
            self.local_name = local_name

    h6147 = g.model_filter("H6147")
    assert h6147(Dev(None), Adv("ihoment_H6147_1A2B"))
    assert h6147(Dev("Govee_h6147_FFEE"), Adv(None))
    assert not h6147(Dev("GBK_H618E_688F"), Adv(None))
    assert not h6147(Dev(None), Adv(None))
    assert g.model_filter("h618e")(Dev("GBK_H618E_688F"), Adv(None))


def test_model_flag_defaults_to_h6147():
    assert g.parse_args([]).model == "H6147"
    assert g.parse_args(["--model", "H618E"]).model == "H618E"


def test_cli_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        g.parse_args(["--variant", "nope"])
