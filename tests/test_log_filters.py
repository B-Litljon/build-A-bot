"""
Tests for core.log_filters — the guard that stopped a weekend of Cloudflare
error pages from writing a 447 MB soak log.

Glossary:
    _record -- builds a bare LogRecord with the given msg/args, so the filter
        is exercised the way logging actually calls it (args unformatted).
    _CLOUDFLARE_PAGE -- a stand-in for the ~96 KB HTML body OANDA's edge proxy
        returns on 502/520; multi-line and far over the cap.
"""

import logging

from core.log_filters import (
    DEFAULT_MAX_CHARS,
    TruncatingFilter,
    install,
)

_CLOUDFLARE_PAGE = "<!DOCTYPE html>\n<html>\n" + ("<div>padding</div>\n" * 5000)


def _record(msg, args=()):
    return logging.LogRecord(
        name="oandapyV20.oandapyV20",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_short_message_is_untouched():
    original = 'request failed [503,{"errorMessage":"System under maintenance"}]'
    rec = _record(original)
    assert TruncatingFilter().filter(rec) is True
    assert rec.getMessage() == original


def test_lazy_args_still_formatted_when_short():
    rec = _record("stream disconnected: %s", ("maintenance",))
    TruncatingFilter().filter(rec)
    assert rec.getMessage() == "stream disconnected: maintenance"


def test_html_body_is_capped_and_flattened():
    rec = _record("request %s failed [520,%s]", ("/candles", _CLOUDFLARE_PAGE))
    assert TruncatingFilter().filter(rec) is True

    out = rec.getMessage()
    assert "\n" not in out, "a flooded log tail must stay one line per record"
    assert len(out) < DEFAULT_MAX_CHARS + 60
    # The diagnostically useful head survives.
    assert out.startswith("request /candles failed [520,<!DOCTYPE html>")
    assert "truncated" in out


def test_truncation_reports_how_much_was_dropped():
    rec = _record("x" * 5000)
    TruncatingFilter(max_chars=100).filter(rec)
    assert "[truncated 4900 chars]" in rec.getMessage()


def test_filter_is_idempotent_across_handlers():
    rec = _record("%s", (_CLOUDFLARE_PAGE,))
    f = TruncatingFilter()
    f.filter(rec)
    first = rec.getMessage()
    f.filter(rec)
    assert rec.getMessage() == first


def test_zero_disables_truncation():
    rec = _record(_CLOUDFLARE_PAGE)
    assert TruncatingFilter(max_chars=0).filter(rec) is True
    assert rec.getMessage() == _CLOUDFLARE_PAGE


def test_broken_format_args_are_passed_through_not_masked():
    """A record whose args don't match its format string must still reach the
    handler, so the underlying bug surfaces instead of being swallowed."""
    rec = _record("needs %d args", ("not-an-int",))
    assert TruncatingFilter().filter(rec) is True


def test_install_attaches_to_every_handler(monkeypatch):
    monkeypatch.delenv("LOG_MAX_CHARS", raising=False)
    logger = logging.getLogger("test_install_target")
    logger.handlers = [logging.NullHandler(), logging.NullHandler()]

    installed = install(logger)

    assert all(installed in h.filters for h in logger.handlers)
    assert installed.max_chars == DEFAULT_MAX_CHARS


def test_install_honours_env_override(monkeypatch):
    monkeypatch.setenv("LOG_MAX_CHARS", "42")
    logger = logging.getLogger("test_install_env")
    logger.handlers = [logging.NullHandler()]
    assert install(logger).max_chars == 42


def test_install_falls_back_on_garbage_env(monkeypatch):
    monkeypatch.setenv("LOG_MAX_CHARS", "banana")
    logger = logging.getLogger("test_install_garbage")
    logger.handlers = [logging.NullHandler()]
    assert install(logger).max_chars == DEFAULT_MAX_CHARS
