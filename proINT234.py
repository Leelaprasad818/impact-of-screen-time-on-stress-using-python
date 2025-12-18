import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

df = pd.read_excel("mental_health_social_media_datasetCA.xlsx")

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df.drop(columns=["person_name", "date"], inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

df["gender"] = LabelEncoder().fit_transform(df["gender"])
df["platform"] = LabelEncoder().fit_transform(df["platform"])
df["mental_state"] = LabelEncoder().fit_transform(df["mental_state"])

print(df.head())



plt.hist(df["daily_screen_time_min"], bins=30, color="tomato")
plt.xlabel("Daily Screen Time (minutes)")
plt.ylabel("Number of Users")
plt.title("Distribution of Daily Screen Time")
plt.grid(True)
plt.show()


plt.scatter(df["daily_screen_time_min"], df["stress_level"], color="purple")
plt.xlabel("Daily Screen Time (minutes)")
plt.ylabel("Stress Level")
plt.title("Daily Screen Time vs Stress Level")
plt.grid(True)
plt.show()

# Chart 3: Average Stress Level by Platform
avg_stress_platform = df.groupby("platform")["stress_level"].mean()
plt.bar(avg_stress_platform.index, avg_stress_platform.values, color="darkorange")
plt.xlabel("Platform (encoded)")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Platform")
plt.grid(True)
plt.show()


mental_counts = df["mental_state"].value_counts().sort_index()
plt.bar(mental_counts.index, mental_counts.values, color="green")
plt.xlabel("Mental State (encoded)")
plt.ylabel("Number of Users")
plt.title("Distribution of Mental States")
plt.grid(True)
plt.show()



print("\n============ REGRESSION MODEL ============")

X_reg = df.drop(columns=["stress_level"])
y_reg = df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

print("\nRegression Performance:")
print("MAE :", round(mean_absolute_error(y_test, y_pred_reg), 3))
print("MSE :", round(mean_squared_error(y_test, y_pred_reg), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_reg)), 3))
print("R²  :", round(r2_score(y_test, y_pred_reg), 3))

plt.scatter(y_test, y_pred_reg, color="red")
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Regression: Actual vs Predicted")
plt.grid(True)
plt.show()


print("\n============ CLASSIFICATION MODEL ============")

X_clf = df.drop(columns=["mental_state"])
y_clf = df["mental_state"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred_clf), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred_clf))

cm = confusion_matrix(y_test, y_pred_clf)
print("\nConfusion Matrix:\n", cm)

plt.imshow(cm, cmap="coolwarm")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\n============ CLUSTERING (K-Means) ============")

scaled_data = StandardScaler().fit_transform(df)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

print("\nCluster Distribution:\n", df["Cluster"].value_counts())


plt.scatter(df["stress_level"], df["anxiety_level"], c=df["Cluster"], cmap="rainbow")
plt.xlabel("Stress Level")
plt.ylabel("Anxiety Level")
plt.title("Clusters: Stress vs Anxiety")
plt.grid(True)
plt.show()
