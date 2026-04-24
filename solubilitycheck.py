import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

print("🦾 QA Pro V3.1: Starting Extreme Precision Training (5000 Iterations)...")

# --- 1. Data Loading ---
df = pd.read_csv('solubility_data.csv')

# --- 2. Feature Extraction ---
def get_advanced_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        # Adding 5 key physical properties for maximum depth
        logp = Descriptors.MolLogP(mol)
        molwt = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        return np.concatenate((fp, [logp, molwt, tpsa, h_donors, h_acceptors]))
    except: return None

df['Features'] = df['SMILES'].apply(get_advanced_features)
df = df.dropna(subset=['Features'])
X = np.stack(df['Features'].values)
y = df['LogS'].values

# Split Data
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.1, random_state=42)

# --- 3. THE 5000 ITERATION ENGINES ---

# Engine 1: The Generalist (Optimized for all zones)
model_stage1 = XGBRegressor(
    n_estimators=5000,      # Huge iteration count
    learning_rate=0.01,     # Very slow, patient learning
    max_depth=8,            # Deep trees to catch complex patterns
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',     # Faster training on large datasets
    random_state=42,
    n_jobs=-1               # Using all your CPU power
)

print("🧠 Training Engine 1... This will take a few minutes. Patience is key.")
model_stage1.fit(
    X_train_all, y_train_all,
    eval_set=[(X_test_all, y_test_all)],
    verbose=100             # Every 100 rounds it will print progress
)

# Engine 2: The Specialist (Targeting Zone A < -6)
zone_a_mask = y < -6
X_zone_a = X[zone_a_mask]
y_zone_a = y[zone_a_mask]
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_zone_a, y_zone_a, test_size=0.1, random_state=42)

print("🕵️‍♂️ Training Engine 2 (The Specialist) for Zone A Precision...")
model_stage2 = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.005,    # Even slower for the toughest chemicals
    max_depth=10,           # Extra depth for Zone A outliers
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_stage2.fit(
    X_train_A, y_train_A,
    eval_set=[(X_test_A, y_test_A)],
    verbose=100
)

# --- 4. Saving the New "Super-Brains" ---
joblib.dump(model_stage1, 'model_stage1_general.pkl')
joblib.dump(model_stage2, 'model_stage2_specialist.pkl')

print("\n" + "⭐"*15)
print("V3.1 EXTREME TRAINING COMPLETE!")
print("Your AI is now calibrated for high-precision laboratory standards.")
print("⭐"*15)