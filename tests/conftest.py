"""Test-process safeguards for the account-free verification suite."""

from __future__ import annotations

import socket


def _network_attempt(*args, **kwargs):
    raise AssertionError(
        "The account-free test suite must not open a network connection. "
        "Mock the provider boundary instead."
    )


# Install the guard while pytest is loading tests, before any test body can
# create a provider client. The process is short-lived, so restoration is not
# needed and this also protects collection-time imports.
socket.create_connection = _network_attempt
socket.socket.connect = _network_attempt
socket.socket.connect_ex = _network_attempt
socket.socket.sendto = _network_attempt
