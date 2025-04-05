---

## **Donor Analytics Report for NCCF: May–January New Donors**
---

## 1. Data Cleaning & Preprocessing

### **Key Actions:**
- Converted `Amount` from string to float.
- Parsed `Date` into `datetime` format.
- Replaced missing values:
  - Categorical → “Unknown”
  - Numeric → 0
- Standardized engagement flags (e.g., subscriber and marker columns).

---

## 2. Exploratory Data Analysis (EDA)

### **Donation Summary:**
- **Average donation:** $170.60
- **Median donation:** $64.50
- **Max donation:** $5,000 (major outlier)

### **Donation Types:**
- **Cash:** 205 entries (most common)
- **Recurring Gift:** 17 entries
- **Gift-in-Kind:** 5 entries

### **Appeals Performance:**
- Most effective appeals by new donor volume:
  1. *24 HQ Pelican ToC*
  2. *24 New Member*
  3. *24 Membership Renew*

### **Origin Source Codes:**
- Most entries use a generic or missing code (`0`)
- Secondary sources: `www.nccoast.org`, `SecurePayAPI`, `subscribe`

### **Engagement:**
- 259 donors (out of 302) did **not submit any surveys**
- Only a small number showed higher engagement (submitted 1–6 surveys)

---

## 3. Donor Clustering

We applied **KMeans Clustering** on donation amount and survey submissions:

### **Clusters Identified:**
| Cluster | Avg Donation | Avg Surveys | Persona |
|---------|---------------|--------------|---------|
| 0       | $125.65       | 0.09         | **Standard Donor** — moderate gifts, low engagement |
| 1       | $5,000.00     | 1.00         | **High-Value Donor** — large one-time donors |
| 2       | $67.50        | 2.61         | **Engaged Micro-Donor** — low gift size, high engagement |

These clusters provide clear **behavioral segmentation** of new donors.

---

## 4. Classification Models

We trained models to **predict donor cluster** using features like amount, engagement, and appeal/source info.

### **Models Used:**
| Model                  | Accuracy (basic) | Accuracy (enhanced) |
|------------------------|------------------|----------------------|
| Random Forest          | 100%             | 95.6%               |
| Logistic Regression    | 98.9%            | 98.9%               |
| Gradient Boosting      | 100%             | 100%              |

###  **Enhanced Features:**
- `Amount`, `Survey Submitted`
- One-hot encodings of `Type`, `Fund(s)`, `Appeal(s)`, `Origin Source Code`

### **Feature Importance:**
- Top features influencing prediction:
  - `Survey Submitted` — strongest predictor of engagement behavior
  - `Amount` — drives identification of high-value donors
  - Encoded appeals and type helped fine-tune persona identification

---

## 5. Donor Journey & Time Trends

We visualized donations over time by cluster:

### **Donation Count Trends:**
- **Cluster 0** (Standard) donors showed consistent month-to-month donations.
- **Cluster 2** (Engaged Micro) peaked at specific periods, indicating event/campaign responses.
- **Cluster 1** (High-Value) appeared only occasionally.

### **Donation Volume Trends:**
- **Cluster 1** spiked donation totals dramatically during their few appearances.
- **Cluster 0/2** offered stability in monthly giving volume.

---

## 6. Donor Personas Summary

| Cluster | Persona Name            | Characteristics |
|---------|-------------------------|-----------------|
| 0       | Standard Donor          | Moderate donation, minimal engagement |
| 1       | High-Value Donor        | Very large donation, limited engagement |
| 2       | Engaged Micro-Donor     | Small donation, frequent engagement |

These personas can guide **targeted outreach**, **appeal design**, and **engagement strategies**.

---

## Strategic Recommendations

1. **Personalized Campaigns**: Leverage personas for targeted messaging.
2. **Boost Engagement**: Use surveys and follow-ups to convert Cluster 0 donors into Cluster 2 behavior.
3. **Track Source Codes**: Improve attribution — many entries lacked meaningful source tracking.
4. **Encourage Recurring Gifts**: Predictable Cluster 2 behavior suggests opportunity for auto-renewals.
5. **Prioritize Top Appeals**: Replicate and enhance successful campaigns like *24 HQ Pelican ToC*.
