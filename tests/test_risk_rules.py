import pytest
from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Minimal transaction that scores exactly 0 — all signals below every threshold."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# Exact score per signal at every threshold boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,expected_score", [
    # device_risk_score: below low / at low / just below high / at high
    ("device_risk_score", 39,  0),
    ("device_risk_score", 40, 10),
    ("device_risk_score", 69, 10),
    ("device_risk_score", 70, 25),
    # is_international: domestic / international
    ("is_international",  0,  0),
    ("is_international",  1, 15),
    # amount_usd: below medium / at medium / just below high / at high
    ("amount_usd",  499,  0),
    ("amount_usd",  500, 10),
    ("amount_usd",  999, 10),
    ("amount_usd", 1000, 25),
    # velocity_24h: below low / at low / just below high / at high
    ("velocity_24h", 2,  0),
    ("velocity_24h", 3,  5),
    ("velocity_24h", 5,  5),
    ("velocity_24h", 6, 20),
    # failed_logins_24h: below low / at low / just below high / at high
    ("failed_logins_24h", 1,  0),
    ("failed_logins_24h", 2, 10),
    ("failed_logins_24h", 4, 10),
    ("failed_logins_24h", 5, 20),
    # prior_chargebacks: none / one / two / more
    ("prior_chargebacks", 0,  0),
    ("prior_chargebacks", 1,  5),
    ("prior_chargebacks", 2, 20),
    ("prior_chargebacks", 3, 20),
])
def test_signal_exact_score(field, value, expected_score):
    """Each signal in isolation produces the exact expected contribution."""
    assert score_transaction(_base_tx(**{field: value})) == expected_score


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

def test_score_clamped_at_100():
    # All signals at maximum: 25+15+25+20+20+20 = 125, must clamp to 100.
    tx = {
        "device_risk_score": 70,
        "is_international": 1,
        "amount_usd": 1000,
        "velocity_24h": 6,
        "failed_logins_24h": 5,
        "prior_chargebacks": 2,
    }
    assert score_transaction(tx) == 100


def test_score_never_negative():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# label_risk exact boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_label", [
    (0,   "low"),
    (29,  "low"),
    (30,  "medium"),
    (59,  "medium"),
    (60,  "high"),
    (100, "high"),
])
def test_label_risk_boundary(score, expected_label):
    assert label_risk(score) == expected_label


# ---------------------------------------------------------------------------
# Realistic scenario tests
# ---------------------------------------------------------------------------

def test_high_risk_transaction_scores_high():
    """Transaction matching all major fraud signals must land in the high-risk band."""
    score = score_transaction({
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1400,
        "velocity_24h": 8,
        "failed_logins_24h": 7,
        "prior_chargebacks": 3,
    })
    assert score >= 60, f"Expected high risk (>=60) but got {score}"


def test_clean_transaction_scores_low():
    """Normal domestic purchase from a trusted device should stay in the low-risk band."""
    assert score_transaction(_base_tx()) < 30


def test_signals_are_additive():
    """Adding more fraud signals should monotonically increase the score."""
    base          = score_transaction(_base_tx())
    with_device   = score_transaction(_base_tx(device_risk_score=80))
    with_intl     = score_transaction(_base_tx(device_risk_score=80, is_international=1))
    with_velocity = score_transaction(_base_tx(device_risk_score=80, is_international=1, velocity_24h=8))
    assert base < with_device < with_intl < with_velocity
