import pytest

from diskovod.security import SecretBox, password_matches


def test_secret_box_rejects_short_keys():
    with pytest.raises(ValueError):
        SecretBox("short")


def test_secret_box_round_trip_and_authentication():
    box = SecretBox("a" * 32)
    encrypted = box.seal("hello")
    assert encrypted != "hello"
    assert box.open(encrypted) == "hello"
    with pytest.raises(Exception):
        box.open(encrypted[:-2] + "aa")


def test_password_comparison():
    assert password_matches("correct", "correct")
    assert not password_matches("wrong", "correct")
