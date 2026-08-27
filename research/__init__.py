"""Research infrastructure: the point-in-time record and the signals built on it.

Deliberately outside `tools/`. Those five servers ship in the homelab image and
answer questions about the world; this package accumulates a private record and
forms opinions from it. Keeping them apart is the same reason `alpaca` is
excluded from the image -- a data-source host should not be able to trade, and
should not need to.
"""
