"""
test_soil_health.py — Verification suite for Soil Health Card (SHC) Scanner
and PAU/ICAR Fertilizer Planning module.
"""
import sys
import io
from pathlib import Path

# Add project root and backend
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

def test_soil_health_service():
    print("\n--- 1. Testing SoilHealthService ---")
    from services.soil_health_service import get_soil_health_service, PUNJAB_SAMPLE_CARDS

    svc = get_soil_health_service()
    assert len(PUNJAB_SAMPLE_CARDS) >= 4, "Expected at least 4 Punjab benchmark sample cards"

    # Test Ludhiana card
    res = svc.parse_soil_health_card(preloaded_key="Ludhiana Central (PAU Research Zone - Loamy Soil)")
    assert res["success"] is True
    assert len(res["records"]) == 12
    print("  [PASS] Parse preloaded Ludhiana SHC card (12 parameters verified)")

    # Test Wheat Fertilizer Plan in Punjabi
    plan_pa = svc.generate_fertilizer_plan(
        crop="Wheat",
        soil_values=res["raw_values"],
        land_acres=2.0,
        language="Punjabi"
    )
    assert plan_pa["success"] is True
    assert len(plan_pa["fertilizers"]) >= 3
    assert len(plan_pa["splits"]) == 3
    assert plan_pa["total_saved_inr"] > 0, "High K soil should trigger Potash cost savings"
    assert "PAU" in plan_pa["citations"][0]
    assert "ਕਣਕ" in plan_pa["explanation"]
    print(f"  [PASS] PAU Wheat fertilizer plan (Punjabi) - Total cost: ₹{plan_pa['total_cost_inr']:,.0f}, Saved: ₹{plan_pa['total_saved_inr']:,.0f}")

    # Test Paddy Fertilizer Plan in Hindi
    plan_hi = svc.generate_fertilizer_plan(
        crop="Paddy",
        soil_values=res["raw_values"],
        land_acres=1.5,
        language="Hindi"
    )
    assert plan_hi["success"] is True
    assert "किसान" in plan_hi["explanation"]
    print(f"  [PASS] PAU Paddy fertilizer plan (Hindi) - Total cost: ₹{plan_hi['total_cost_inr']:,.0f}")

    # Test Potato Plan in English
    plan_en = svc.generate_fertilizer_plan(
        crop="Potato",
        soil_values=res["raw_values"],
        land_acres=1.0,
        language="English"
    )
    assert plan_en["success"] is True
    assert "Acre" in plan_en["explanation"]
    print(f"  [PASS] PAU Potato fertilizer plan (English) - Total cost: ₹{plan_en['total_cost_inr']:,.0f}")


def test_agent_tool():
    print("\n--- 2. Testing Farming Agent Soil Tool ---")
    from agent.farming_agent import run_agent

    ans = run_agent(
        "गेहूं की फसल के लिए सॉइल हेल्थ कार्ड अनुसार खाद की सिफारिश क्या है?",
        language="Hindi",
        location="Ludhiana"
    )
    assert len(ans["response"]) > 100
    assert "soil_health_card_fertilizer" in ans.get("sources", [])
    print("  [PASS] Agent successfully routes soil fertilizer queries to soil_health_card_fertilizer tool")


def test_streamlit_app():
    print("\n--- 3. Testing Streamlit AppTest for All 7 Tabs ---")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    at.run()
    if at.exception:
        print("  [FAIL] Streamlit AppTest exceptions encountered:")
        for exc in at.exception:
            print(f"    - {exc}")
        sys.exit(1)
    else:
        print("  [PASS] Streamlit AppTest loaded all 7 tabs with 0 exceptions!")


if __name__ == "__main__":
    test_soil_health_service()
    test_agent_tool()
    test_streamlit_app()
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!\n")
