#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["bleak>=0.22"]
# ///
"""Set a Govee H6147 LED strip to solid red over BLE, then turn it off.

Usage:
    uv run govee_h6147.py                        # interactive: confirm which color packet works
    uv run govee_h6147.py --scan                 # list nearby Govee devices, touch nothing
    uv run govee_h6147.py --variant rgbic-1501   # send one specific color packet
    uv run govee_h6147.py --address <ADDR>       # skip name matching (MAC on Linux/Windows,
                                                 # CoreBluetooth UUID on macOS)

Govee BLE packet format: 20 bytes = header byte (0x33 for commands, 0xAA for
keep-alive/status) + command + payload, zero padded to 19 bytes, then one XOR
checksum byte over the first 19.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Sequence

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakError

# Govee's GATT characteristics (same on basically every Govee BLE light).
WRITE_CHAR = "00010203-0405-0607-0809-0a0b0c0d2b11"
NOTIFY_CHAR = "00010203-0405-0607-0809-0a0b0c0d2b10"

DEFAULT_MODEL = "H6147"
GOVEE_NAME_HINTS = ("GOVEE", "IHOMENT", "GBK_", "MINGER", "H6", "H7")

PACKET_LEN = 20
HEADER_COMMAND = 0x33
HEADER_KEEPALIVE = 0xAA

CMD_POWER = 0x01
CMD_COLOR = 0x05

RED = (0xFF, 0x00, 0x00)
KEEPALIVE_INTERVAL = 2.0
WATCHDOG_GRACE = 10.0  # extra seconds past the scan timeout before we call it a hang
SETTLE_DELAY = 0.8  # time for the strip to apply a packet before we ask about it


# --------------------------------------------------------------------------- packets


def build_packet(header: int, payload: Sequence[int]) -> bytes:
    """Return a 20-byte Govee packet: header + payload, zero padded, XOR checksum last."""
    body = bytes([header, *payload])
    if len(body) > PACKET_LEN - 1:
        raise ValueError(
            f"payload too long: {len(payload)} bytes (max {PACKET_LEN - 2})"
        )
    body = body.ljust(PACKET_LEN - 1, b"\x00")
    checksum = 0
    for byte in body:
        checksum ^= byte
    return body + bytes([checksum])


def power_packet(on: bool) -> bytes:
    return build_packet(HEADER_COMMAND, [CMD_POWER, 0x01 if on else 0x00])


def keepalive_packet() -> bytes:
    return build_packet(HEADER_KEEPALIVE, [0x01])


# Segment block used by RGBIC "set segment color" packets:
# 5 bytes of white/colour-temperature fields (unused here, zero) followed by a
# little-endian segment bitmask. 0xFF 0x7F selects segments 0-14 = every
# segment on 15-segment strips; extra bits are ignored by shorter strips.
RGBIC_ALL_SEGMENTS = (0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x7F)


@dataclass(frozen=True)
class ColorVariant:
    name: str
    description: str
    mode: tuple[int, ...]  # bytes between CMD_COLOR and the RGB triple
    suffix: tuple[int, ...] = ()  # bytes after the RGB triple

    def packet(self, rgb: tuple[int, int, int]) -> bytes:
        return build_packet(HEADER_COMMAND, [CMD_COLOR, *self.mode, *rgb, *self.suffix])


# Tried in order. Plain RGB modes first, RGBIC segment modes as fallback.
VARIANTS: tuple[ColorVariant, ...] = (
    ColorVariant("rgb-02", "RGB manual mode 0x02 (classic H61xx strips)", (0x02,)),
    ColorVariant("rgb-0d", "RGB manual mode 0x0D (newer RGB firmware)", (0x0D,)),
    ColorVariant(
        "rgbic-1501",
        "RGBIC segment colour 0x15 0x01, all segments",
        (0x15, 0x01),
        RGBIC_ALL_SEGMENTS,
    ),
    ColorVariant(
        "rgbic-0b",
        "RGBIC segment colour 0x0B, all segments (older RGBIC firmware)",
        (0x0B,),
        RGBIC_ALL_SEGMENTS,
    ),
)
VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}


# --------------------------------------------------------------------------- BLE helpers


BLUETOOTH_HANG_HINT = (
    "Bluetooth did not respond. On macOS, give your terminal app Bluetooth access "
    "(System Settings > Privacy & Security > Bluetooth) and make sure Bluetooth is on."
)


async def with_watchdog(coro, timeout: float):
    """bleak can block forever if the OS Bluetooth stack never becomes ready."""
    try:
        return await asyncio.wait_for(coro, timeout + WATCHDOG_GRACE)
    except asyncio.TimeoutError:
        raise SystemExit(BLUETOOTH_HANG_HINT) from None


def _adv_name(device: BLEDevice, adv: AdvertisementData | None) -> str:
    return (adv.local_name if adv and adv.local_name else None) or device.name or ""


def model_filter(model: str):
    """Match Govee names like "ihoment_H6147_1A2B" / "Govee_H6147_1A2B" / "GBK_H6147_1A2B"."""
    needle = model.upper()

    def matches(device: BLEDevice, adv: AdvertisementData) -> bool:
        return needle in _adv_name(device, adv).upper()

    return matches


async def scan_and_list(model: str, timeout: float) -> None:
    print(f"Scanning {timeout:.0f}s for BLE devices...")
    found = await with_watchdog(
        BleakScanner.discover(timeout=timeout, return_adv=True), timeout
    )
    rows = []
    for device, adv in found.values():
        name = _adv_name(device, adv)
        if any(hint in name.upper() for hint in GOVEE_NAME_HINTS):
            rows.append((adv.rssi, name, device.address))
    if not rows:
        print(f"No Govee-looking devices among {len(found)} seen.")
        return
    for rssi, name, address in sorted(rows, reverse=True):
        marker = "  <-- target" if model.upper() in name.upper() else ""
        print(f"{rssi:>5} dBm  {name:<28} {address}{marker}")


async def find_strip(address: str | None, model: str, timeout: float) -> BLEDevice:
    if address:
        print(f"Looking for {address} ({timeout:.0f}s)...")
        search = BleakScanner.find_device_by_address(address, timeout=timeout)
    else:
        print(f"Scanning for a device named *{model}* ({timeout:.0f}s)...")
        search = BleakScanner.find_device_by_filter(model_filter(model), timeout=timeout)
    device = await with_watchdog(search, timeout)
    if device is None:
        raise SystemExit(
            "Strip not found. Check: strip powered, within ~5 m, Govee phone app closed "
            "(the strip accepts only one BLE connection), and this terminal has "
            "Bluetooth permission. Try --scan to see what is advertising."
        )
    return device


async def ask_yes_no(prompt: str) -> bool:
    """input() on a daemon thread so Ctrl-C never hangs waiting for Enter."""
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[str] = loop.create_future()

    def reader() -> None:
        try:
            line = input(prompt)
        except EOFError:
            line = ""
        loop.call_soon_threadsafe(lambda: answer.done() or answer.set_result(line))

    threading.Thread(target=reader, daemon=True).start()
    return (await answer).strip().lower() in {"y", "yes"}


class GoveeStrip:
    def __init__(self, client: BleakClient, verbose: bool) -> None:
        self.client = client
        self.verbose = verbose
        self._lock = asyncio.Lock()  # keep-alive and commands must not interleave writes

    async def send(self, packet: bytes, label: str) -> None:
        async with self._lock:
            if self.verbose:
                print(f"  -> {label:<12} {packet.hex(' ')}")
            await self.client.write_gatt_char(WRITE_CHAR, packet)

    async def keepalive_loop(self) -> None:
        # Govee strips drop idle BLE links; the official app pings every ~2 s.
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            with suppress(BleakError):
                await self.send(keepalive_packet(), "keep-alive")

    def on_notify(self, _sender: object, data: bytearray) -> None:
        if self.verbose:
            print(f"  <- notify       {bytes(data).hex(' ')}")


async def apply_red(strip: GoveeStrip, forced: str | None, interactive: bool) -> ColorVariant | None:
    if forced:
        variant = VARIANTS_BY_NAME[forced]
        await strip.send(variant.packet(RED), variant.name)
        print(f"Sent red using {variant.name}: {variant.description}")
        return variant

    if not interactive:
        # No way to see the strip: send every variant. Firmware ignores modes it
        # does not implement, so the strip ends up red if any of them is valid.
        for variant in VARIANTS:
            await strip.send(variant.packet(RED), variant.name)
            await asyncio.sleep(0.3)
        print("Non-interactive: sent all colour variants (unsupported ones are ignored).")
        return VARIANTS[-1]

    for variant in VARIANTS:
        print(f"Trying {variant.name}: {variant.description}")
        await strip.send(variant.packet(RED), variant.name)
        await asyncio.sleep(SETTLE_DELAY)
        if await ask_yes_no("  Is the strip solid red now? [y/N] "):
            print(f"  {variant.name} works. Next time: --variant {variant.name}")
            return variant
    return None


async def run(args: argparse.Namespace) -> int:
    device = await find_strip(args.address, args.model, args.timeout)
    print(f"Found {device.name or '?'} [{device.address}], connecting...")

    async with BleakClient(device, timeout=20.0) as client:
        strip = GoveeStrip(client, args.verbose)
        with suppress(BleakError):
            await client.start_notify(NOTIFY_CHAR, strip.on_notify)
        keepalive = asyncio.create_task(strip.keepalive_loop())
        try:
            await strip.send(power_packet(True), "power on")
            await asyncio.sleep(0.5)

            interactive = sys.stdin.isatty() and not args.yes
            variant = await apply_red(strip, args.variant, interactive)
            if variant is None:
                print(
                    "No variant produced solid red. Run with --verbose and check the "
                    "notify replies; the strip may need a model-specific packet."
                )
                return 1

            print(f"Holding red for {args.hold:g}s...")
            await asyncio.sleep(args.hold)
            return 0
        finally:
            keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive
            if client.is_connected:
                with suppress(BleakError):
                    await strip.send(power_packet(False), "power off")
                    print("Strip off.")
                    await asyncio.sleep(0.3)  # let the write flush before disconnect


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scan", action="store_true", help="list nearby Govee devices and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model to match in the BLE name (default {DEFAULT_MODEL})")
    parser.add_argument("--address", help="connect to this address instead of matching by name")
    parser.add_argument("--variant", choices=list(VARIANTS_BY_NAME), help="send only this colour packet")
    parser.add_argument("--yes", "-y", action="store_true", help="never prompt; send all variants")
    parser.add_argument("--hold", type=float, default=5.0, help="seconds to stay red (default 5)")
    parser.add_argument("--timeout", type=float, default=15.0, help="scan timeout seconds (default 15)")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every packet sent/received")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.scan:
            asyncio.run(scan_and_list(args.model, args.timeout))
            return 0
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except BleakError as exc:
        print(f"Bluetooth error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
