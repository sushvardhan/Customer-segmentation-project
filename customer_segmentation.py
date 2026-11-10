"""
Customer Segmentation Project
Run: python src/customer_segmentation.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = [
    "age", "annual_income_k", "spending_score", "purchases_last_12m",
    "avg_order_value", "days_since_last_purchase",
    "website_visits_monthly", "discount_usage_rate"
]
X = df[features]

if X.isnull().any().any():
    X = X.fillna(X.median())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

scores = {}
for k in range(2, 7):
    candidate = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = candidate.fit_predict(Xs)
    scores[k] = silhouette_score(Xs, labels)

best_k = max(scores, key=scores.get)
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["segment"] = model.fit_predict(Xs)

profile = df.groupby("segment")[features].mean().round(2)
profile["customer_count"] = df["segment"].value_counts().sort_index()
profile.to_csv(OUT / "segment_profile.csv")
df.to_csv(OUT / "customer_segments.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar([str(k) for k in scores], list(scores.values()))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("K selection comparison")
plt.tight_layout()
plt.savefig(OUT / "k_selection.png", dpi=160)
plt.close()

coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df["segment"], s=18, alpha=0.7)
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title(f"Customer segments (K-Means, k={best_k})")
plt.colorbar(scatter, label="Segment")
plt.tight_layout()
plt.savefig(OUT / "customer_segments_pca.png", dpi=160)
plt.close()

print(f"Completed: {len(df)} customers segmented into {best_k} groups.")
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()})
print("\\nSegment profile:\\n", profile)
print(f"Files saved to: {OUT}")
