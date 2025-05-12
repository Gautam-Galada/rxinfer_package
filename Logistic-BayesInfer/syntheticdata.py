import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic features
age = np.random.normal(loc=35, scale=10, size=n_samples)               # continuous
income = np.random.normal(loc=50000, scale=15000, size=n_samples)      # continuous
gender = np.random.choice(["male", "female"], size=n_samples)          # categorical
education = np.random.choice(["highschool", "bachelor", "master"], size=n_samples)  # categorical

# Encode as features for label generation
gender_bin = (gender == "male").astype(int)
education_score = [0 if e == "highschool" else 1 if e == "bachelor" else 2 for e in education]

# Simulate true weights
logits = (
    0.05 * age +
    0.0004 * income +
    0.7 * gender_bin +
    1.2 * np.array(education_score) -
    25  # bias term
)

# Generate binary labels using sigmoid + Bernoulli
prob = 1 / (1 + np.exp(-logits))
target = np.random.binomial(1, prob)

# Assemble DataFrame
df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Gender": gender,
    "Education": education,
    "Target": target
})

# Save to CSV
csv_path = "synthetic_classification_data.csv"
df.to_csv(csv_path, index=False)

csv_path
