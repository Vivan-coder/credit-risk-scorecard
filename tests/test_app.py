"""End-to-end tests: run the real Streamlit script headlessly and drive its widgets."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def fresh():
    return AppTest.from_file(APP, default_timeout=180).run()


def banner(at):
    """Which decision banner is showing: 'Approve' (success), 'Review' (warning) or 'Decline' (error)."""
    found = [(kind, el.value) for kind, els in (("Approve", at.success), ("Review", at.warning), ("Decline", at.error)) for el in els if el.value.startswith("###")]
    assert len(found) == 1, found
    assert found[0][1].startswith("### " + found[0][0])
    return found[0][0]


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_load_has_no_exceptions_and_three_tabs():
    at = fresh()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Score an applicant", "Portfolio view", "Model card"]
    assert banner(at) == "Review"                                   # default example is the borderline one


@pytest.mark.parametrize("example,expected", [("Low risk (approve)", "Approve"), ("Borderline (review)", "Review"), ("High risk (decline)", "Decline")])
def test_examples_land_in_the_intended_decision(example, expected):
    at = fresh()
    at.selectbox(key="example").select(example).run()
    assert not at.exception
    assert banner(at) == expected
    pd_text = metric(at, "Probability of default"); score = float(metric(at, "Credit score"))
    pd_val = float(pd_text.strip("%")) / 100
    assert (expected == "Approve") == (pd_val < 0.05) and (expected == "Decline") == (pd_val >= 0.10)
    assert 430 <= score <= 630


def test_worse_inputs_never_improve_the_score():
    at = fresh(); at.selectbox(key="example").select("Low risk (approve)").run()
    base = float(metric(at, "Credit score"))
    at.number_input(key="in_l90").set_value(2).run()
    assert not at.exception and float(metric(at, "Credit score")) < base
    at.slider(key="in_util").set_value(150).run()
    assert not at.exception and float(metric(at, "Credit score")) < base


def test_income_unknown_and_zero_income_do_not_crash():
    at = fresh()
    at.checkbox(key="in_income_unknown").check().run()
    assert not at.exception and banner(at) in {"Approve", "Review", "Decline"}
    at.checkbox(key="in_income_unknown").uncheck().run()
    at.number_input(key="in_income").set_value(0).run()
    assert not at.exception


def test_manual_loan_amount_changes_expected_loss_linearly():
    at = fresh()
    at.checkbox(key="use_est_ead").uncheck().run()
    at.number_input(key="in_loan").set_value(10_000.0).run()
    el1 = float(metric(at, "Expected loss").replace("BHD", "").replace(",", ""))
    at.number_input(key="in_loan").set_value(20_000.0).run()
    el2 = float(metric(at, "Expected loss").replace("BHD", "").replace(",", ""))
    assert not at.exception and el2 == pytest.approx(2 * el1, rel=0.01)


def test_extreme_but_valid_inputs():
    at = fresh()
    at.number_input(key="in_age").set_value(100).run(); assert not at.exception
    at.number_input(key="in_age").set_value(18).run(); assert not at.exception
    for k in ("in_l30", "in_l60", "in_l90"):
        at.number_input(key=k).set_value(20).run()
    at.slider(key="in_util").set_value(200).run()
    assert not at.exception and banner(at) == "Decline"


def test_portfolio_tab_slider_is_monotone():
    at = fresh()
    sl = at.slider[-1]
    sl.set_value(5.0).run(); a_low = float(metric(at, "Approval rate").strip("%"))
    at.slider[-1].set_value(30.0).run(); a_high = float(metric(at, "Approval rate").strip("%"))
    assert not at.exception and a_high > a_low


def test_portfolio_model_switch():
    at = fresh()
    at.radio[0].set_value("LightGBM (challenger)").run()
    assert not at.exception


def test_model_card_numbers_come_from_reports():
    at = fresh()
    text = " ".join(m.value for m in at.markdown)
    assert "10%" in text and "6.68%" in text and "Not Bahraini data" in text and "No reject inference" in text
