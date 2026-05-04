# E-commerce-Funnel-and-Customer-Behavior-Analysis
E-commerce funnel and customer behavior analysis using SQL and Python

## Problem

The business needed to understand why users were browsing the catalog but not purchacing.

## Approach

I cleaned and normalized raw Shopify and sales data, calculated core purchase and funnel metrics, built dashboards of KPI metrics, and performed exploratory analysis to identify product discovery issues and prepare the analytical foundation for a future recommendation system.

## Key Finding

The largest funnel drop was found at the view→cart stage (~13%), indicating friction at the product discovery step rather than at checkout.

## Next Step

This finding motivated the [further recommendation system project](https://github.com/Ichwour/Personalized-Recommendation-System-ML-Pipeline), focused on improving product selection through personalized candidate ranking.

## Data Extraction & Cleaning Demo

The raw Shopify export contained many technical and unstructured fields.  
For analysis, I selected only the product attributes needed for customer behavior and sales analysis.

```sql
SELECT
    "Variant SKU" AS product_id,
    "Title" AS product_name,
    "Vendor" AS vendor,
    "Product Category" AS product_category,
    "Variant Price" AS product_price,
    "Variant Inventory Qty" AS inventory_qty,
    "Product Country (product.metafields.product.country)" AS country,
    "Product Region (product.metafields.product.region)" AS region,
    "Wine Sugar (product.metafields.wine.sugar)" AS sugar,
    "Wine Varietals (product.metafields.wine.varietals)" AS varietals,
    "Product Year (product.metafields.product.year)" AS product_year,
    "Wine Rating (product.metafields.wine.rating)" AS rating
FROM products
WHERE "Variant SKU" IS NOT NULL;
```

The next step was to connect sales history with product metadata and create an analysis-ready event table.

```sql
SELECT
    e."Client ID" AS client_id,
    e."Session ID" AS session_id,
    e."Date" AS event_time,
    e."Event Type" AS event_type,
    e.product_id,
    p.product_name,
    p.vendor,
    p.product_category,
    p.product_price,
    p.country,
    p.region,
    p.sugar,
    p.varietals,
    p.product_year,
    p.rating,
    e."Price" AS price
FROM events AS e
LEFT JOIN products AS p
    ON e.product_id = p.product_id;
```
    
After joining the datasets, I performed basic validation and cleaning in pandas before saving the normalized analytical dataset.

```python
import pandas as pd

df = pd.read_csv("event_products.csv")
print(df.shape)
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))
key_cols = ["client_id", "session_id", "event_time", "event_type", "product_id", "price"]
df = df.dropna(subset=key_cols)
text_cols = ["event_type", "country", "region", "sugar", "varietals", "vendor"]

for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": None, "": None})
        )

df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
numeric_cols = ["product_price", "price", "product_year", "rating"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["event_time", "price"])
df.to_csv("data.csv", index=False)
print("Clean dataset saved:", df.shape)
```

## KPI Calculation: Average Order Value (AOV)

AOV was calculated from purchase events using item-level transaction prices.  
Since one session can contain multiple purchased products, order value was reconstructed by summing product prices within each purchase session.

```sql
SELECT
    AVG(order_sum) AS AOV
FROM (
    SELECT
        session_id,
        SUM(price) AS order_sum
    FROM data
    WHERE event_type = 'purchase'
    GROUP BY session_id
) t;
```

Funnel conversion rates were calculated at the session level using unique session identifiers for each stage of the user journey.

```sql
WITH counts AS (
    SELECT
        COUNT(DISTINCT CASE WHEN event_type = 'view' THEN session_id END) AS view_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN session_id END) AS cart_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN session_id END) AS purchase_sessions
    FROM data
)
SELECT
    view_sessions,
    cart_sessions,
    purchase_sessions,
    cart_sessions * 1.0 / view_sessions AS view_to_cart,
    purchase_sessions * 1.0 / cart_sessions AS cart_to_purchase
FROM counts;
```

- view → cart: ~13%
- cart → purchase: ~40%
- AOV: ~180 euro

## Dashboard

The cleaned dataset was visualized in Tableau to provide a business-facing view of the funnel and key purchase metrics.

Dashboard included:
- KPI cards: AOV, purchase sessions, revenue
- Funnel: view → cart → purchase
- Price distribution and demand heatmap

## Business Outcome

The analysis identified the main funnel bottleneck at the view→cart stage (~13% conversion).
This showed that the main issue was not checkout completion, but product discovery and selection.
Based on this insight, the next analytical direction was to improve product discovery through recommendation-based candidate selection.

## Additional pre-analysis for Recommendation System
## Candidate-Level Feature Analysis

To understand which features separate true purchases from baseline candidates, I analyzed the candidate-level dataset generated from the baseline retrieval space.

Each event contained:
- 1 true purchased item (`label = 1`)
- 999 baseline candidate negatives (`label = 0`)

Total dataset size:
- Events: **6,643**
- Rows: **6,643,000**
- Positives: **6,643**
- Negatives: **6,636,357**

The analysis was performed using:
- **Chi-square test** for categorical product attributes
- **Kolmogorov–Smirnov test** for numeric feature distributions
- **Pearson correlation** for numeric feature association with the target label

```python
from scipy.stats import chi2_contingency, ks_2samp, pearsonr

contingency = pd.crosstab(df["vendor"], df["label"])
chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
positive_price = df[df["label"] == 1]["price"].dropna()
negative_price = df[df["label"] == 0]["price"].dropna()
ks_stat, ks_p = ks_2samp(positive_price, negative_price)
corr, corr_p = pearsonr(df["profile_score"], df["label"])
```

### Results

The strongest numeric separation was observed in **price**:

- True item mean price: **32.56**
- Candidate mean price: **68.15**
- True item median price: **18.00**
- Candidate median price: **30.00**
- KS statistic: **0.238**

This indicates that true purchases were generally shifted toward lower-priced items compared to the baseline candidate pool.

The second useful numeric signal was **profile_score**:

- True item mean profile_score: **0.347**
- Candidate mean profile_score: **0.283**
- KS statistic: **0.151**
- Pearson correlation: **0.0135**

The Pearson correlation was weak due to the highly imbalanced structure of the dataset (1 positive vs 999 negatives per event), but the distribution shift confirmed that profile similarity still carried useful signal.

Categorical features also showed statistically significant separation, but with weak effect size:

- `vendor`: Cramer's V ≈ **0.035**
- `varietals_1`: Cramer's V ≈ **0.029**
- `varietals_2`: Cramer's V ≈ **0.028**
- `region`: Cramer's V ≈ **0.025**
- `country`: Cramer's V ≈ **0.011**
- `color`: Cramer's V ≈ **0.006**
- `sugar`: Cramer's V ≈ **0.005**

These results suggest that product metadata contains signal, but no single categorical feature strongly separates true purchases from candidate alternatives.

### Key Problem: Weak Separation in the Candidate Space

The analysis showed a relatively flat feature-separation picture.
This was expected: the candidate pool was already generated by a baseline recommender, meaning that negatives were not random catalog items but relatively plausible alternatives. As the candidate space becomes narrower and more relevant, true items and negatives naturally become more similar.

This created a hard-negative problem:
- obvious mismatches were already removed by retrieval
- remaining candidates had similar metadata
- single features had weak separation
- ranking required combining many weak signals

