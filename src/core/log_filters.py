"""
log_filters.py — keep a single log record from swallowing the disk
==================================================================

The live bot logs broker errors verbatim. That is fine when the broker
answers with its own JSON (``{"errorMessage": "System under maintenance"}``),
and catastrophic when it answers with an edge-proxy HTML page: OANDA sits
behind Cloudflare, and a 502/520 during weekend maintenance returns a ~96 KB
styled error page. Three call sites log that body — ``oandapyV20``'s own
logger, ``data.oanda_provider.get_historical_bars``, and the orchestrator's
"stream disconnected" — and the reconnect loop retries roughly once a minute.

Measured on the 2026-08-16 soak: 4,640 HTML dumps, 909k lines, **447 MB**
in one log file, of which 99% was Cloudflare markup.

:class:`TruncatingFilter` caps any single record and flattens it to one line,
so an unreadable blob becomes a readable one-liner that still shows the URL
and the HTTP status. It is attached to the root *handler* rather than to a
logger so it also covers third-party libraries we do not control.

Glossary:
    TruncatingFilter -- logging.Filter that shortens over-long records in
        place. Never drops a record; only rewrites its text.
    max_chars -- cap on the formatted message, in characters. Records at or
        under it pass through completely untouched.
    DEFAULT_MAX_CHARS -- 1500. Comfortably fits any real broker JSON error
        plus the request URL, while cutting a Cloudflare page to its first
        useful line.
    LOG_MAX_CHARS -- env override for the cap. Set it to 0 to disable
        truncation entirely (full-fidelity debugging).
    _TRUNCATION_MARKER -- suffix appended to a shortened message, carrying the
        number of characters removed so the loss is visible, not silent.
    _TRUNCATED_FLAG -- attribute stamped on a record once it has been
        shortened. Makes the filter idempotent when a record fans out to
        several handlers, which would otherwise chop the marker itself.
"""

from __future__ import annotations

import logging
import os

DEFAULT_MAX_CHARS = 1500
_TRUNCATION_MARKER = "… [truncated {dropped} chars]"
_TRUNCATED_FLAG = "_log_filters_truncated"


class TruncatingFilter(logging.Filter):
    """
    Shorten over-long log records to a single capped line.

    Attach to a *handler* so records from third-party loggers are covered
    too::

        for handler in logging.getLogger().handlers:
            handler.addFilter(TruncatingFilter())

    The record is rewritten in place: ``msg`` becomes the already-formatted,
    truncated text and ``args`` is cleared, which keeps the change idempotent
    if the record passes through more than one handler.
    """

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        super().__init__()
        self.max_chars = max_chars

    def filter(self, record: logging.LogRecord) -> bool:
        if self.max_chars <= 0:
            return True

        # Already handled by an earlier handler's filter. Without this the
        # marker itself pushes the record back over the cap and gets chopped
        # on every subsequent pass.
        if getattr(record, _TRUNCATED_FLAG, False):
            return True

        try:
            message = record.getMessage()
        except Exception:
            # A record whose args do not match its format string is already
            # broken; let the handler surface that rather than masking it.
            return True

        if len(message) <= self.max_chars and "\n" not in message:
            return True

        # Flatten first: a 200-line HTML blob is unreadable in a log tail even
        # when it fits the cap.
        flattened = " ".join(message.split())

        if len(flattened) > self.max_chars:
            dropped = len(flattened) - self.max_chars
            flattened = flattened[: self.max_chars] + _TRUNCATION_MARKER.format(
                dropped=dropped
            )

        record.msg = flattened
        record.args = ()
        setattr(record, _TRUNCATED_FLAG, True)
        return True


def install(
    logger: logging.Logger | None = None, max_chars: int | None = None
) -> TruncatingFilter:
    """
    Attach a :class:`TruncatingFilter` to every handler on ``logger``.

    Defaults to the root logger, and to ``LOG_MAX_CHARS`` from the
    environment (falling back to :data:`DEFAULT_MAX_CHARS`). Returns the
    filter so callers can tune or remove it.
    """
    target = logger if logger is not None else logging.getLogger()
    if max_chars is None:
        try:
            max_chars = int(os.getenv("LOG_MAX_CHARS", DEFAULT_MAX_CHARS))
        except ValueError:
            max_chars = DEFAULT_MAX_CHARS

    log_filter = TruncatingFilter(max_chars)
    for handler in target.handlers:
        handler.addFilter(log_filter)
    return log_filter
