"""
explain_stub.py

STUB for the eventual explanation-generation layer (Day 6, not yet built).
Returns a fixed placeholder string so daily_integration_check.py can run
end-to-end before the real explain() exists.

CRITICAL: nothing returned by this stub is a real explanation.
"""


def explain(context: dict) -> str:
    """STUB. Real signature/behavior TBD (Day 6)."""
    return f"[STUB EXPLANATION] dominant_deferral_signal={context.get('dominant_deferral_signal')!r}"


if __name__ == "__main__":
    result = explain({"dominant_deferral_signal": "gap", "object_id": "apple"})
    print(f"Stub explain() returned: {result!r}")
