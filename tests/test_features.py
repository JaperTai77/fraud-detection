import pandas as pd
import pytest
from features import build_model_frame


def _txn_df(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 42,
        "timestamp": "2026-01-01 12:00:00",
        "amount_usd": 100.0,
        "merchant_category": "grocery",
        "channel": "web",
        "device_risk_score": 10,
        "ip_country": "US",
        "is_international": 0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "chargeback_within_60d": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _acct_df(**overrides):
    row = {
        "account_id": 42,
        "customer_name": "Test User",
        "country": "US",
        "signup_date": "2024-01-01",
        "kyc_level": "full",
        "account_age_days": 365,
        "prior_chargebacks": 0,
        "is_vip": "N",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Merge behaviour
# ---------------------------------------------------------------------------

def test_account_fields_merged_into_output():
    result = build_model_frame(_txn_df(), _acct_df(customer_name="Alice", prior_chargebacks=2))
    assert result.iloc[0]["customer_name"] == "Alice"
    assert result.iloc[0]["prior_chargebacks"] == 2


def test_unknown_account_id_produces_nan():
    """Transactions with no matching account should not raise; account columns become NaN."""
    result = build_model_frame(_txn_df(account_id=9999), _acct_df(account_id=42))
    assert pd.isna(result.iloc[0]["customer_name"])


def test_multiple_transactions_all_enriched():
    txns = pd.DataFrame([
        {**_txn_df().iloc[0].to_dict(), "transaction_id": 1, "account_id": 42, "amount_usd": 200},
        {**_txn_df().iloc[0].to_dict(), "transaction_id": 2, "account_id": 42, "amount_usd": 1500},
    ])
    result = build_model_frame(txns, _acct_df())
    assert len(result) == 2
    assert list(result["is_large_amount"]) == [0, 1]


# ---------------------------------------------------------------------------
# is_large_amount boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected_flag", [
    (999.99, 0),
    (1000.0, 1),
    (1000.1, 1),
    (5000.0, 1),
    (0.0,    0),
])
def test_is_large_amount_boundary(amount, expected_flag):
    result = build_model_frame(_txn_df(amount_usd=amount), _acct_df())
    assert result.iloc[0]["is_large_amount"] == expected_flag


# ---------------------------------------------------------------------------
# login_pressure bins  (-1,0]→none  (0,2]→low  (2,100]→high
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failed_logins,expected_pressure", [
    (0, "none"),
    (1, "low"),
    (2, "low"),
    (3, "high"),
    (10, "high"),
])
def test_login_pressure_bins(failed_logins, expected_pressure):
    result = build_model_frame(_txn_df(failed_logins_24h=failed_logins), _acct_df())
    assert str(result.iloc[0]["login_pressure"]) == expected_pressure


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "transaction_id", "account_id", "amount_usd", "device_risk_score",
    "is_international", "velocity_24h", "failed_logins_24h",
    "prior_chargebacks", "is_large_amount", "login_pressure",
}

def test_all_required_columns_present():
    result = build_model_frame(_txn_df(), _acct_df())
    assert EXPECTED_COLUMNS.issubset(set(result.columns))
