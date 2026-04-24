import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import os

# --- 1. Page Config ---
st.set_page_config(page_title="QA Pro V3: Dual-Engine Predictor", page_icon="🧪", layout="wide")

# --- 2. Load Both Engines ---
@st.cache_resource
def load_engines():
    m1 = joblib.load('model_stage1_general.pkl')
    m2 = joblib.load('model_stage2_specialist.pkl')
    return m1, m2

model_gen, model_spec = load_engines()

# --- 3. Sidebar & UI ---
st.sidebar.header("🔬 V3.0 Dual-Engine Controls")
temp = st.sidebar.slider("Temperature (°C)", 0, 100, 25)

st.title("🧪 Smart Chemical Solubility Dashboard 3.0")
st.markdown("**Experimental Dual-Engine Logic:** Using Fingerprints + LogP + MolWt for Zone A Precision.")

smiles_input = st.text_input("Enter SMILES String:", "CCCCCCCCCCCCCCCC") # Hexadecane default

# --- 4. Prediction Logic ---
if st.button("Run Deep Analysis", type="primary"):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Draw.MolToImage(mol, size=(400, 300)), caption="Molecular Structure")

       # --- V3.1 Updated Feature Extraction ---
fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
logp = Descriptors.MolLogP(mol)
molwt = Descriptors.MolWt(mol)
tpsa = Descriptors.TPSA(mol)
h_donors = Descriptors.NumHDonors(mol) # 🟢 Naya Feature
h_acceptors = Descriptors.NumHAcceptors(mol) # 🟢 Naya Feature

# Ab total 1029 features combine honge
features = np.concatenate((fp, [logp, molwt, tpsa, h_donors, h_acceptors])).reshape(1, -1)
        # STEP 1: General Engine Prediction
        raw_val = model_gen.predict(features)[0]
        
        final_val = raw_val
        zone_label = ""
        is_specialist_used = False

        # STEP 2: Logic Switch
        if raw_val < -6:
            # Switch to Specialist Engine 2
            is_specialist_used = True
            final_val = model_spec.predict(features)[0]
            
            # Micro-Stratification
            if -7 <= final_val < -6: zone_label = "Zone A1 (Very Low)"
            elif -8 <= final_val < -7: zone_label = "Zone A2 (Extreme Low)"
            elif -9 <= final_val < -8: zone_label = "Zone A3 (Ultra Low)"
            else: zone_label = "Zone A4 (Inert/Non-Soluble)"
        else:
            if -6 <= raw_val < -4: zone_label = "Zone B (Low)"
            elif -4 <= raw_val < -2: zone_label = "Zone C (Medium)"
            else: zone_label = "Zone D (High)"

        with col2:
            st.subheader("📊 Deep Analysis Results")
            st.metric("Final Predicted LogS", f"{final_val:.3f}")
            
            if is_specialist_used:
                st.warning(f"🔍 **Specialist Engine Active:** {zone_label}")
            else:
                st.success(f"✅ **General Engine Active:** {zone_label}")

            with st.expander("📝 Technical QA Audit"):
                st.write(f"- **Engine 1 Raw:** `{raw_val:.3f}`")
                st.write(f"- **Engine 2 Corrected:** `{final_val:.3f}`")
                st.write(f"- **LogP (Oil Affinity):** `{logp:.2f}`")
                st.write(f"- **Molecular Weight:** `{molwt:.2f}`")

    else:
        st.error("❌ Invalid SMILES")

st.caption("Developed by Ankush | V3.0 Dual-Engine System")