import pytest

from evalrag.observability.tracer import trace_span


def test_trace_span_returns_value():
    @trace_span("demo")
    def f(x):
        return x * 2
    assert f(3) == 6


def test_trace_span_emits_event(capsys):
    @trace_span("demo")
    def f(x):
        return x + 1
    f(2)
    out = capsys.readouterr()
    assert "demo" in (out.out + out.err)


def test_trace_span_propagates_exceptions():
    @trace_span("demo")
    def f():
        raise ValueError("boom")
    with pytest.raises(ValueError):
        f()
