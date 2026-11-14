"""Offline tests for light_shows (no Bluetooth). Run:
uv run --with-requirements requirements.txt --with pytest pytest -q
"""

import pytest

import light_shows as ls
from govee_h6147 import VARIANTS_BY_NAME, brightness_packet, power_packet

COLOR = VARIANTS_BY_NAME["rgb-02"].packet
BASE = COLOR((255, 0, 0))
ON, OFF = power_packet(True), power_packet(False)


def director(**kwargs) -> ls.LightDirector:
    return ls.LightDirector(COLOR, (255, 0, 0), hold=10.0, **kwargs)


def packets(steps) -> list[bytes]:
    return [packet for _label, packet in steps]


# ------------------------------------------------------------------ packets


def test_brightness_packet_scales_percent_to_the_strip_range():
    assert brightness_packet(100, 255)[:3] == bytes([0x33, 0x04, 255])
    assert brightness_packet(50, 100)[:3] == bytes([0x33, 0x04, 50])
    assert brightness_packet(0, 255)[2] == 1  # never fully dark: that is what "off" is for
    assert brightness_packet(250, 255)[2] == 255  # clamped


# -------------------------------------------------------------------- shows


@pytest.mark.parametrize("name", sorted(ls.SHOWS))
def test_every_show_gives_valid_colours_for_its_whole_length(name):
    show = ls.SHOWS[name]
    assert show.duration > 0
    seen = set()
    steps = int(show.duration * 20)
    for i in range(steps):
        rgb = show.color(i / 20)
        assert len(rgb) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)
        seen.add(rgb)
    assert len(seen) > 1  # it actually animates


def test_rainbow_does_not_ask_for_a_new_colour_every_millisecond():
    show = ls.SHOWS["rainbow"]
    colours = {show.color(i / 1000) for i in range(1000)}
    assert len(colours) <= 20  # one second: stays under what Bluetooth can keep up with


def test_blink_alternates_colour_and_dark():
    show = ls.blink("x", (0, 255, 0), times=2, phase=0.2)
    assert show.color(0.05) == (0, 255, 0)
    assert show.color(0.25) == (0, 0, 0)
    assert show.color(0.45) == (0, 255, 0)
    assert show.duration == pytest.approx(0.8)


# ----------------------------------------------------------------- director


def test_motion_turns_the_light_on_then_off_after_the_hold():
    d = director()
    assert d.step(0.0) == []  # nothing happened yet: leave the strip alone
    d.motion(1.0)
    assert packets(d.step(1.0)) == [ON, BASE]
    assert d.light_on is True and d.last_change == 1.0
    assert d.step(5.0) == []
    assert packets(d.step(11.5)) == [OFF]
    assert d.light_on is False and d.last_change == 11.5


def test_show_plays_then_switches_off_when_nobody_is_moving():
    d = director()
    d.play("disarmed")
    first = d.step(0.0)
    assert packets(first)[0] == ON and len(first) == 2
    assert d.light_on is True and d.busy
    assert d.step(0.01) == []  # same colour: nothing to send
    assert packets(d.step(0.25)) == [COLOR((0, 0, 0))]
    end = ls.SHOWS["disarmed"].duration
    assert packets(d.step(end + 0.01)) == [OFF]
    assert d.light_on is False and not d.busy
    assert d.step(end + 1) == []


def test_show_hands_back_to_the_motion_colour_when_motion_was_recent():
    d = director()
    d.motion(0.0)
    d.step(0.0)
    d.play("disarmed")
    d.step(0.1)
    end = 0.1 + ls.SHOWS["disarmed"].duration
    assert packets(d.step(end + 0.01)) == [BASE]
    assert d.light_on is True
    assert packets(d.step(10.5)) == [OFF]  # and the normal hold still applies


def test_every_show_write_counts_as_a_lighting_change():
    d = director()
    d.play("disarmed")
    d.step(3.0)
    d.step(3.25)
    assert d.last_change == 3.25


def test_a_new_show_replaces_the_running_one():
    d = director()
    d.play("disarmed")
    d.step(0.0)
    d.play("armed")
    assert packets(d.step(0.05)) == [COLOR(ls.SHOWS["armed"].color(0.0))]


def test_unknown_show_is_refused():
    d = director()
    assert d.play("nope") is False
    assert d.step(0.0) == [] and not d.busy


def test_off_stops_a_show_and_the_light():
    d = director()
    d.motion(0.0)
    d.play("rainbow")
    d.step(0.0)
    d.off()
    assert packets(d.step(0.5)) == [OFF]
    assert d.light_on is False and not d.busy
    assert d.step(1.0) == []  # the earlier motion does not bring it back


def test_brightness_turns_the_light_on_and_sends_the_level():
    d = director(brightness_max=255)
    d.set_brightness(40, 2.0)
    assert packets(d.step(2.0)) == [ON, BASE, brightness_packet(40, 255)]
    assert d.step(2.05) == []
    assert d.brightness == 40


def test_brightness_writes_are_rate_limited_and_keep_only_the_latest():
    d = director(brightness_max=255)
    d.set_brightness(40, 0.0)
    d.step(0.0)
    d.set_brightness(50, 0.02)
    d.set_brightness(60, 0.04)
    assert d.busy
    assert d.step(0.04) == []  # too soon after the last write
    assert packets(d.step(0.2)) == [brightness_packet(60, 255)]
    assert not d.busy


def test_dialling_brightness_keeps_the_light_on():
    d = director()
    d.set_brightness(40, 0.0)
    d.step(0.0)
    d.set_brightness(45, 9.0)
    d.step(9.0)
    assert d.step(12.0) == []  # hold counts from the last touch, not the first
    assert packets(d.step(19.5)) == [OFF]


def test_shutdown_turns_a_lit_strip_off_once():
    d = director()
    assert d.shutdown() == []
    d.motion(0.0)
    d.step(0.0)
    assert packets(d.shutdown()) == [OFF]
    assert d.shutdown() == []


def test_snooze_show_fades_out_and_leaves_the_light_off_despite_recent_motion():
    show = ls.SHOWS["snooze"]
    assert show.color(0.0) == (0, 80, 255)
    assert sum(show.color(show.duration * 0.9)) < sum(show.color(show.duration * 0.3)) < sum(show.color(0.0))
    d = director()
    d.motion(0.0)
    d.step(0.0)
    d.play("snooze")
    d.step(0.1)
    assert packets(d.step(0.2 + show.duration)) == [OFF]
    assert d.light_on is False
    assert d.step(0.3 + show.duration) == []  # the earlier motion does not bring it back


def test_toggle_switches_on_and_stays_on_without_motion():
    d = director()
    assert d.toggle() is True and d.busy
    assert packets(d.step(0.0)) == [ON, BASE]
    assert not d.busy
    assert d.step(500.0) == []  # no motion for ages: still on
    assert d.toggle() is False
    assert packets(d.step(501.0)) == [OFF]
    assert d.step(502.0) == [] and d.light_on is False


def test_toggle_switches_off_a_light_that_motion_turned_on():
    d = director()
    d.motion(0.0)
    d.step(0.0)
    assert d.toggle() is False
    assert packets(d.step(1.0)) == [OFF]
    assert d.step(2.0) == []  # the earlier motion does not bring it back


def test_toggle_stops_a_running_show():
    d = director()
    d.play("rainbow")
    d.step(0.0)
    assert d.toggle() is False
    assert packets(d.step(0.5)) == [OFF]


def test_a_show_hands_back_to_a_light_that_was_toggled_on():
    d = director()
    d.toggle()
    d.step(0.0)
    d.play("disarmed")
    d.step(100.0)
    assert packets(d.step(100.1 + ls.SHOWS["disarmed"].duration)) == [BASE]
    assert d.light_on is True
