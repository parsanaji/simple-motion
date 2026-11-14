"""Light shows and the "what should the strip do now" logic for motion_detect.

Everything here is plain arithmetic: no Bluetooth, no sleeping. motion_detect's
GoveeController asks LightDirector.step() what to send and writes the packets.

Add a show by putting one more entry in SHOWS: a name, a length in seconds and
a function that gives the colour at each moment.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Callable

from govee_h6147 import brightness_packet, power_packet

RGB = tuple[int, int, int]
Step = tuple[str, bytes]  # (label for the log, 20-byte packet)

SHOW_RATE = 12.0  # colour changes per second a show may ask for
BRIGHTNESS_WRITE_SECONDS = 0.1  # at most one brightness packet this often
BLACK: RGB = (0, 0, 0)


@dataclass(frozen=True)
class Show:
    name: str
    duration: float  # seconds
    color: Callable[[float], RGB]  # seconds since the show started -> colour
    ends_dark: bool = False  # True: the light stays off afterwards, even if motion was recent


def _tick(t: float, rate: float = SHOW_RATE) -> float:
    """Snap time to the show rate so a show never asks for more than `rate` colours a second."""
    return int(t * rate) / rate


def rainbow(name: str = "rainbow", duration: float = 12.0, cycle: float = 3.0) -> Show:
    def color(t: float) -> RGB:
        r, g, b = colorsys.hsv_to_rgb((_tick(t) / cycle) % 1.0, 1.0, 1.0)
        return (round(r * 255), round(g * 255), round(b * 255))

    return Show(name, duration, color)


def heartbeat(name: str = "heartbeat", rgb: RGB = (255, 20, 90), duration: float = 10.0, beat: float = 1.2) -> Show:
    """Two quick pulses, then a rest, like a heart."""

    def color(t: float) -> RGB:
        phase = (_tick(t) % beat) / beat
        if phase < 0.15:
            level = 1.0
        elif phase < 0.3:
            level = 0.25
        elif phase < 0.45:
            level = 0.8
        else:
            level = max(0.08, 0.8 * (1 - (phase - 0.45) / 0.55) ** 2)
        return (round(rgb[0] * level), round(rgb[1] * level), round(rgb[2] * level))

    return Show(name, duration, color)


DISCO_COLOURS: tuple[RGB, ...] = (
    (255, 0, 0), (0, 80, 255), (255, 200, 0), (255, 0, 200), (0, 255, 80), (140, 0, 255), (0, 220, 255),
)


def disco(name: str = "disco", duration: float = 10.0, step: float = 0.3) -> Show:
    def color(t: float) -> RGB:
        return DISCO_COLOURS[(int(t / step) * 3) % len(DISCO_COLOURS)]  # *3: neighbours differ a lot

    return Show(name, duration, color)


def blink(name: str, rgb: RGB, times: int = 2, phase: float = 0.2) -> Show:
    def color(t: float) -> RGB:
        return rgb if int(t / phase) % 2 == 0 else BLACK

    return Show(name, times * 2 * phase, color)


def fade(name: str, rgb: RGB, duration: float = 3.0) -> Show:
    """Start at full colour and dim to dark, like dozing off."""

    def color(t: float) -> RGB:
        level = max(0.0, 1 - _tick(t) / duration) ** 2
        return (round(rgb[0] * level), round(rgb[1] * level), round(rgb[2] * level))

    return Show(name, duration, color, ends_dark=True)


SHOWS: dict[str, Show] = {
    show.name: show
    for show in (
        rainbow(),
        heartbeat(),
        disco(),
        blink("armed", (255, 0, 0)),
        blink("disarmed", (0, 255, 0)),
        fade("snooze", (0, 80, 255)),
    )
}


def light_action(light_on: bool | None, last_motion: float | None, now: float, hold: float) -> str | None:
    """Decide what the strip should do. light_on None = unknown (never touched)."""
    if last_motion is None:
        return None
    want_on = now - last_motion < hold
    if want_on and light_on is not True:
        return "on"
    if not want_on and light_on is True:
        return "off"
    return None


class LightDirector:
    """Decides which packets the strip needs next.

    motion() / play() / set_brightness() / off() / toggle() may be called from another
    thread: each only stores a value. step() does the deciding and is called
    by whoever owns the Bluetooth link.
    """

    def __init__(
        self,
        color_packet: Callable[[RGB], bytes],
        base_color: RGB,
        hold: float,
        brightness_max: int = 255,
    ) -> None:
        self._color_packet = color_packet
        self._base_packet = color_packet(base_color)
        self.hold = hold
        self.brightness_max = brightness_max
        self.light_on: bool | None = None  # None = never touched
        self.last_change: float | None = None  # when the room lighting last changed
        self.brightness: int | None = None  # percent last sent
        self._last_motion: float | None = None
        self._pending_show: Show | None = None
        self._pending_off = False
        self._latched = False  # switched on by hand: stays on until switched off
        self._show: tuple[Show, float] | None = None  # running show and its start time
        self._show_rgb: RGB | None = None
        self._want_brightness: int | None = None
        self._brightness_at = float("-inf")

    # ---- requests (any thread)
    def motion(self, now: float) -> None:
        self._last_motion = now

    def play(self, name: str) -> bool:
        show = SHOWS.get(name)
        if show is None:
            return False
        self._pending_show = show
        return True

    def set_brightness(self, percent: float, now: float) -> None:
        self._want_brightness = max(1, min(100, round(percent)))
        self._last_motion = now  # the light comes on, and stays on while you dial

    def off(self) -> None:
        self._pending_off = True

    def toggle(self) -> bool:
        """Flip the light by hand. Returns True if it is now meant to be on.

        On this way it stays on, with or without motion, until off() or the next toggle().
        """
        if self.light_on is True or self._show or self._pending_show:
            self.off()
            return False
        self._latched = True
        return True

    @property
    def busy(self) -> bool:
        """True while step() should be called often (a show or a brightness change is under way)."""
        return bool(
            self._show or self._pending_show or self._pending_off
            or (self._latched and self.light_on is not True)
            or (self._want_brightness is not None and self._want_brightness != self.brightness)
        )

    # ---- deciding (Bluetooth thread)
    def step(self, now: float) -> list[Step]:
        steps: list[Step] = []
        if self._pending_off:
            self._pending_off = self._latched = False
            self._pending_show = self._show = self._show_rgb = None
            self._last_motion = None
            self._want_brightness = self.brightness
            if self.light_on:
                steps.append(("off", power_packet(False)))
                self.light_on = False
        elif self._pending_show or self._show:
            steps = self._show_step(now)
        else:
            action = light_action(self.light_on, self._last_motion, now, self.hold)
            if self._latched:
                action = "on" if self.light_on is not True else None
            if action == "on":
                steps += [("on", power_packet(True)), ("colour", self._base_packet)]
                self.light_on = True
            elif action == "off":
                steps.append(("off", power_packet(False)))
                self.light_on = False

        if (
            self.light_on
            and self._want_brightness is not None
            and self._want_brightness != self.brightness
            and now - self._brightness_at >= BRIGHTNESS_WRITE_SECONDS
        ):
            self.brightness = self._want_brightness
            self._brightness_at = now
            steps.append((f"brightness {self.brightness}%", brightness_packet(self.brightness, self.brightness_max)))
        if steps:
            self.last_change = now
        return steps

    def _show_step(self, now: float) -> list[Step]:
        if self._pending_show:
            self._show, self._pending_show, self._show_rgb = (self._pending_show, now), None, None
        show, started = self._show
        steps: list[Step] = []
        if now - started < show.duration:
            if self.light_on is not True:
                steps.append(("on", power_packet(True)))
                self.light_on = True
            rgb = show.color(now - started)
            if rgb != self._show_rgb:
                self._show_rgb = rgb
                steps.append((f"show {show.name}", self._color_packet(rgb)))
            return steps
        self._show = self._show_rgb = None
        if show.ends_dark:
            self._last_motion = None
        if self._latched or (self._last_motion is not None and now - self._last_motion < self.hold):
            return [("colour", self._base_packet)]  # back to the motion colour
        self.light_on = False
        return [("off", power_packet(False))]

    def shutdown(self) -> list[Step]:
        """Packets to send on exit: never leave the strip glowing."""
        self._show = self._pending_show = None
        if not self.light_on:
            return []
        self.light_on = False
        return [("off", power_packet(False))]
