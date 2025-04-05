## 1. **Data Cleaning and Preprocessing**

### **Raw Dataset Overview**
The original CSV dataset consisted of 302 rows and 22 columns representing new donors, their donation behavior, and multiple categorical attributes tied to marketing channels, engagement signals, and donation metadata.

---

### **Python Tools Used**
- `pandas`: for loading, inspecting, cleaning, and manipulating data.
- `numpy`: for numerical handling.
- `datetime` & `pandas.to_datetime()`: for transforming date strings into `datetime64` objects.

---

###  **Preprocessing Steps**

| Column            | Operation                                  | Tool Used                       |
|-------------------|---------------------------------------------|----------------------------------|
| `Amount`          | Removed "$" and "," → converted to float   | `df['Amount'].replace().astype()` |
| `Date`            | Converted to `datetime` object             | `pd.to_datetime(errors='coerce')` |
| Categorical Nulls | Filled with `"Unknown"`                    | `df.fillna('Unknown')`            |
| Numeric Nulls     | Filled with `0`                            | `df.fillna(0)`                    |

This allowed us to normalize and standardize the data for downstream **EDA, clustering, and modeling**.

---

## 2. **Exploratory Data Analysis (EDA)**

###  **Python Libraries Used**
- `matplotlib.pyplot` and `seaborn`: For visual analytics of distributions and categorical data relationships.

###  **EDA Methods**

####  **Donation Distribution**
```python
sns.histplot(df_cleaned["Amount"], bins=30, kde=True)
```
- **Histogram** + **Kernel Density Estimate (KDE)** to evaluate the skewness of donation amounts.
- Revealed a **right-skewed distribution** (many small donors, few large donors).

####  **Boxplot by Type**
```python
sns.boxplot(x="Type", y="Amount", data=df_cleaned)
```
- Used to identify the **range, quartiles, and outliers** for different donation types.
- Insight: **Recurring donations** had low variability, whereas **Gift-in-Kind** had large outliers.

####  **Categorical Value Counts**
```python
df['Appeal(s)'].value_counts().head(10)
```
- Provided most frequent campaign appeals linked to first-time donations.

#### **Statistical Summaries**
```python
df.describe()  # for mean, std, min, quartiles
```
- For each numeric field: calculated **mean, std dev, median, quartiles, and range**.

---

## 3. **Clustering Analysis (Unsupervised Learning)**

### **Goal**
Segment donors into distinct behavioral groups using unsupervised machine learning.

---

### **Tools Used**
- `scikit-learn (sklearn.cluster.KMeans)`
- `StandardScaler`: Feature normalization
- `PCA (Principal Component Analysis)`: 2D projection for visualization

---

### **Clustering Pipeline**
```python
# Normalize data
X_scaled = StandardScaler().fit_transform(X)

# Reduce dimensions
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Fit KMeans
kmeans = KMeans(n_clusters=3).fit(X_scaled)
```

#### **Science Behind It:**
- **KMeans**: Groups data by minimizing intra-cluster variance using Euclidean distance.
- **PCA**: Eigen decomposition to project high-dimensional data into lower dimensions while preserving variance.
- Used only `Amount` and `Survey Submitted` for simplicity and interpretability.

---

### **Cluster Centroid Results**
Revealed three distinct clusters:
1. High-dollar, low-survey
2. Low-dollar, high-survey
3. Mid-dollar, low-engagement

Used:
```python
df.groupby('Cluster')[['Amount', 'Survey Submitted']].mean()
```

---

## 4. **Classification Modeling (Supervised Learning)**

### **Goal**
Build models that predict donor cluster from input features, for real-time persona detection.

---

### **Models Used**
1. **Random Forest Classifier**
   - Bagging ensemble of decision trees
   - Captures **nonlinear relationships** and **interactions**
2. **Logistic Regression**
   - Linear classifier using **maximum likelihood estimation**
   - Benchmark for interpretability
3. **Gradient Boosting**
   - Ensemble method that builds trees **sequentially** to minimize loss
   - Strong predictive performance for small tabular datasets

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
```

### **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
Used **30% test data** for evaluation.

### **Evaluation Metrics**
```python
classification_report()
accuracy_score()
confusion_matrix()
```

---

## 5. **Feature Engineering**

### Tools
- `pandas.get_dummies()` to encode:
  - `Appeal(s)`
  - `Fund(s)`
  - `Type`
  - `Origin Source Code`

These were merged into the main feature matrix:
```python
X = pd.concat([df[['Amount', 'Survey Submitted']], encoded_categoricals], axis=1)
```

This enhanced model accuracy by **capturing categorical effects**.

---

## 6. **Feature Importance Analysis**

### Random Forest & Gradient Boosting
```python
rf.feature_importances_
gb.feature_importances_
```
- **Survey Submitted** was the most predictive feature.
- `Amount` followed closely.
- Certain appeals and origin sources added explanatory power.

Visualized via:
```python
pd.Series(importances).sort_values().plot(kind='barh')
```

---

## 7. **Donor Journey & Time Series Analysis**

### Goal
Reveal monthly trends in:
- New donor volume
- Total giving behavior

---

### Tools & Science
- `groupby(['Month', 'Cluster'])` to aggregate donations
- Converted `datetime` column into monthly periods:
```python
df['Month'] = df['Date'].dt.to_period('M')
```
- Reconverted to `datetime`:
```python
df['Month'] = df['Month'].dt.to_timestamp()
```
- Plotted using `seaborn.lineplot()` for each cluster

### Insights:
- High-value donors are **seasonal** (possibly campaign-driven)
- Micro-donors are **event-responsive**
- Cluster 0 provides **baseline consistency**

---

## 8. **Donor Persona Creation**

Using clustering results + feature profiles, we created:
- **Standard Donor**: baseline, predictable support
- **High-Value Donor**: fewer in number, strategic targets
- **Engaged Micro-Donor**: frequent survey activity, high conversion potential

These personas can inform **email campaigns**, **renewal strategies**, and **donor nurturing paths**.

---

## Conclusion

### 🔬 **Data Science Summary**
| Phase             | Tool/Method                       | Outcome                             |
|------------------|-----------------------------------|-------------------------------------|
| Data Wrangling    | `pandas`, `numpy`                 | Cleaned dataset                     |
| EDA               | `seaborn`, `describe`, `groupby`  | Understood donor behavior           |
| Clustering        | `KMeans`, `PCA`                   | Segmented donor types               |
| Classification    | `RandomForest`, `LogReg`, `GBM`   | Predict donor persona               |
| Feature Analysis  | `feature_importances_`            | Explained prediction decisions      |
| Time Series       | `groupby().sum()`, `lineplot()`   | Tracked temporal behavior patterns  |
