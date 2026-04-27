from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
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


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_high_device_risk_adds_risk():
    low_device = score_transaction(_base_tx(device_risk_score=10))
    high_device = score_transaction(_base_tx(device_risk_score=80))
    assert high_device > low_device, "High device risk score should increase the fraud score"


def test_international_adds_risk():
    domestic = score_transaction(_base_tx(is_international=0))
    international = score_transaction(_base_tx(is_international=1))
    assert international > domestic, "International transactions should increase the fraud score"


def test_high_velocity_adds_risk():
    low_vel = score_transaction(_base_tx(velocity_24h=1))
    high_vel = score_transaction(_base_tx(velocity_24h=8))
    assert high_vel > low_vel, "High transaction velocity should increase the fraud score"


def test_prior_chargebacks_add_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one_cb = score_transaction(_base_tx(prior_chargebacks=1))
    repeat = score_transaction(_base_tx(prior_chargebacks=3))
    assert one_cb > clean, "One prior chargeback should increase the fraud score"
    assert repeat > one_cb, "More prior chargebacks should produce a higher fraud score"


def test_high_risk_transaction_scores_high():
    """A transaction matching all major fraud signals should reach the high-risk band."""
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
    """A normal domestic transaction from a trusted device should stay low risk."""
    score = score_transaction(_base_tx())
    assert score < 30, f"Expected low risk (<30) but got {score}"
