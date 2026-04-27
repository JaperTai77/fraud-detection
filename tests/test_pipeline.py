"""
Integration tests against the real CSV data files.
These tests verify end-to-end business outcomes, not implementation details.
"""
import pytest
import pandas as pd
from analyze_fraud import load_inputs, score_transactions, summarize_results

CONFIRMED_FRAUD_IDS = {50003, 50006, 50008, 50011, 50013, 50014, 50015, 50019}


@pytest.fixture(scope="module")
def pipeline_output():
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    return scored, summary


# ---------------------------------------------------------------------------
# Score and label validity
# ---------------------------------------------------------------------------

def test_risk_scores_within_valid_range(pipeline_output):
    scored, _ = pipeline_output
    assert scored["risk_score"].between(0, 100).all(), "All risk scores must be in [0, 100]"


def test_risk_labels_are_valid(pipeline_output):
    scored, _ = pipeline_output
    valid = {"low", "medium", "high"}
    unexpected = set(scored["risk_label"].unique()) - valid
    assert not unexpected, f"Unexpected risk labels: {unexpected}"


def test_every_transaction_is_scored(pipeline_output):
    scored, _ = pipeline_output
    assert scored["risk_score"].notna().all()
    assert scored["risk_label"].notna().all()


# ---------------------------------------------------------------------------
# Fraud detection quality
# ---------------------------------------------------------------------------

def test_no_confirmed_fraud_in_low_risk_band(pipeline_output):
    """No transaction that resulted in a chargeback should be labelled low risk."""
    scored, _ = pipeline_output
    fraud_rows = scored[scored["transaction_id"].isin(CONFIRMED_FRAUD_IDS)]
    low_risk_fraud = fraud_rows[fraud_rows["risk_label"] == "low"]
    assert low_risk_fraud.empty, (
        f"Confirmed fraud transactions labelled low risk: "
        f"{low_risk_fraud['transaction_id'].tolist()}"
    )


def test_all_confirmed_fraud_detected(pipeline_output):
    """Every known chargeback transaction must be scored high or medium risk."""
    scored, _ = pipeline_output
    fraud_rows = scored[scored["transaction_id"].isin(CONFIRMED_FRAUD_IDS)]
    assert len(fraud_rows) == len(CONFIRMED_FRAUD_IDS), "Some fraud transactions missing from scored output"
    assert set(fraud_rows["risk_label"].unique()).issubset({"high", "medium"})


def test_high_risk_band_chargeback_rate_is_100_pct(pipeline_output):
    _, summary = pipeline_output
    high = summary[summary["risk_label"] == "high"]
    assert not high.empty, "No transactions scored high risk"
    assert float(high["chargeback_rate"].iloc[0]) == 1.0, (
        f"Expected 100% chargeback rate in high band, got {high['chargeback_rate'].iloc[0]:.0%}"
    )


def test_low_risk_band_chargeback_rate_is_zero(pipeline_output):
    _, summary = pipeline_output
    low = summary[summary["risk_label"] == "low"]
    assert not low.empty, "No transactions scored low risk"
    assert float(low["chargeback_rate"].iloc[0]) == 0.0, (
        f"Expected 0% chargeback rate in low band, got {low['chargeback_rate'].iloc[0]:.0%}"
    )


# ---------------------------------------------------------------------------
# Summary output schema and aggregation correctness
# ---------------------------------------------------------------------------

def test_summary_has_expected_columns(pipeline_output):
    _, summary = pipeline_output
    expected = {"risk_label", "transactions", "total_amount_usd", "avg_amount_usd",
                "chargebacks", "chargeback_rate"}
    assert expected.issubset(set(summary.columns))


def test_summary_transaction_counts_add_up(pipeline_output):
    scored, summary = pipeline_output
    assert summary["transactions"].sum() == len(scored)


def test_summary_chargeback_counts_add_up(pipeline_output):
    _, summary = pipeline_output
    assert summary["chargebacks"].sum() == len(CONFIRMED_FRAUD_IDS)


def test_avg_amount_consistent_with_total(pipeline_output):
    _, summary = pipeline_output
    for _, row in summary.iterrows():
        expected_avg = row["total_amount_usd"] / row["transactions"]
        assert abs(row["avg_amount_usd"] - expected_avg) < 0.01, (
            f"avg_amount_usd inconsistent for {row['risk_label']} band"
        )
