import streamlit as st
import joblib
import pandas as pd

# ---- Compatibility Patch ----
# Fixes version mismatch for internal class dropped in local scikit-learn
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list): 
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
# ------------------------------

# Page config
st.set_page_config(page_title="Startup Predictor", layout="centered")
st.title("🚀 Startup Success Predictor")
st.write("Adjust the metrics below to test the probability of an IPO or Acquisition.")

# Model loading
@st.cache_resource
def load_pipeline():
    return joblib.load("startup_pipeline.pkl")

pipeline = load_pipeline()

# UI Layout: Numeric Features
st.subheader("📊 Company Metrics")
col1, col2 = st.columns(2)

with col1:
    funding_rounds = st.slider("Funding Rounds", min_value=0, max_value=10, value=2)
    team_size = st.slider("Team Size", min_value=0, max_value=1000, value=200)
    burn_rate = st.number_input("Burn Rate (Thousands $)", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
    revenue = st.number_input("Revenue ($)", min_value=0.0, max_value=10000000.0, value=800000.0, step=10000.0)

with col2:
    founder_exp = st.slider("Founder Experience (Years)", min_value=0, max_value=100, value=10)
    traction = st.slider("Product Traction (Users)", min_value=0, max_value=1000000, value=300000, step=10000)
    market_size = st.number_input("Market Size (Billions $)", min_value=0.0, max_value=10000.0, value=30.0, step=1.0)

# UI Layout: Categorical Mappings & Selectboxes
st.subheader("🏢 Categorical Details")

investor_options = {
    "Tier 1 VC": "tier1_vc",
    "Tier 2 VC": "tier2_vc",
    "Angel Investor": "angel",
    "None / Bootstrapped": "none"
}

founder_bg_options = {
    "Academic": "academic",
    "First-Time Founder": "first_time",
    "Ex-Big Tech": "ex_bigtech",
    "Serial Founder": "serial_founder"
}

# Sector labels match closely, but listing them cleanly
sector_options = ["Health", "Fintech", "SaaS", "Ecommerce", "Climate", "Crypto", "AI"]

selected_investor_label = st.selectbox("Primary Investor Type", list(investor_options.keys()))
selected_sector = st.selectbox("Industry Sector", sector_options)
selected_founder_bg_label = st.selectbox("Founder Background", list(founder_bg_options.keys()))

# Inference logic
st.divider()
if st.button("Calculate Success Likelihood", type="primary"):
    
    # Map friendly UI names back to raw model keys
    investor_type = investor_options[selected_investor_label]
    founder_bg = founder_bg_options[selected_founder_bg_label]
    
    # Structure features into expected DataFrame format
    user_data = pd.DataFrame([{
        "funding_rounds": funding_rounds,
        "founder_experience_years": founder_exp,
        "team_size": team_size,
        "market_size_billion": market_size,
        "product_traction_users": traction,
        "burn_rate_million": burn_rate,
        "revenue_million": revenue,
        "investor_type": investor_type,
        "sector": selected_sector,
        "founder_background": founder_bg
    }])
    
    # Run pipeline prediction
    prediction = pipeline.predict(user_data)[0]
    probability = pipeline.predict_proba(user_data)[0][1]
    
    # Output display
    if prediction:
        st.success(f"🎉 **High Probability of Success!** The model estimates a **{probability:.2%}** chance of an IPO or Acquisition.")
    else:
        st.warning(f"⚠️ **High Risk Profile.** The model estimates a **{probability:.2%}** chance of an IPO or Acquisition based on these features.")