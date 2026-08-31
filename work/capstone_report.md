# **Capstone Report — Refresh / Content Opportunity Scoring**

---

- **Author:** Nirvik K.C. (FlyRank ML Intern)
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Nirvik-49/Week-1-FlyRank-AI-Assignment
- **Date:** August 31, 2026

## **0. Abstract**

---

Is there a way of accurately predicting which of the web pages with many impressions and poor performance could get the most clicks if they were refreshed? This investigation used DuckDB and a data snapshot with 120,223 content pages taken from the FlyRank ML Internship warehouse dataset. We have consolidated the Search Console metrics for 90 days and metadata about the content. Then we fitted a gradient-boosted decision tree classifier against a benchmark historical rule that identifies pages simply based on a very low CTR. The resulting model showed an ROC-AUC of 0.814, which is a relatively 25% increase compared to the baseline's 0.651 AUC on a group stratified client holdout split. This technique will give an automatically ranked list of content refresh activities with the explanations that will allow FlyRank editorial teams to focus on the highest-yield content revisions.

## **1. Problem framing**

---

- **Unit of Analysis**: Individual content page (content_hash_id) for rolling daily and 90-day Search Console windows.
- **Output**: A continuously updated refresh priority score (S) that can range from 0 to 1 with the machine-picked reason (example: "High Impression / Low CTR", "Striking Distance Decay", "metadata mismatch").
- **Human Action**: A FlyRank content editor uses the ranked action engine to place certain pages into the position queue for title tag/meta description changes, structural content refreshes, or search intent realignment.
- **Cost of Wrong Call**: The waste of editor man-hours due to false alarm situations that lead to the tagging and revision of content which might have been already fine or is affected by off-siteseasonality; and missed out high-opportunity organic traffic in the false negatives not appearing on pages 1-2.
- **Why Data/ML Helps**: A detailed assessment of 10,000 manual pages is not a feasible task, while a deep learning model can capture the complex relationships among average page rank, impression spread, query diversity, and click-through ratio fluctuation.

## 2. Data safety

---

- **Tables Used**: fact_daily_sample (Search Console daily aggregates), dim_content (page metadata), and fact_query_90d (query-level distributions).
- **Features and Column Drops & Possible Leakage**: Extremely careful with the data leakage issues, and the only thing done was to remove the direct trend labels (trend_direction, trend_pct) and the future performance windows. The client identifiers (client_hash_id) were used only for splitting the group and were Later removed before model training.
- **Data Privacy**: Raw query strings, client names, domain names, target URLs, and user credentials have all been hashed or omitted. No claims about Google's internal algorithm or the refresh causal effect.

## 3. Baseline

---

- **Baseline Definition**: A rule-based benchmark flagging pages where overall_ctr falls below the 25th percentile for their respective integer mean_position bin (Position 1–3, 4–10, 11–20), combined with total_impressions > 500.
- **Fairness & Metrics**: Evaluated on the exact same group-split test dataset and target definition as the ML pipeline.
    - **Baseline Precision@Top-100**: 52.0%
    - **Baseline ROC-AUC**: 0.651
    - **Baseline Base Rate (Target Class)**: 18.4%

## 4. Model / analysis

---

- **Method**: binary classifier with LightGBM, which was trained on binary logarithmic loss and early stopping based on the minimum loss achieved on the validation subset to prevent overfitting.
- **Target Proxy Definition**: A page is considered a high-value refresh opportunity (target=1) if it has an impression volume greater than the 75th percentile in total_impressions and, at the same time, it is in the group that has the minimum value for expected position-adjusted CTR, yet the rank should not change much (mean_position ≤ 20).
- **Features Included**:
    - **log_impressions**: Log-transformed 90-day GSC impressions.
    - **mean_position**: Average Search Console ranking position.
    - **ctr_position_gap**: Difference between observed CTR and position-expected benchmark CTR.
    - **query_count**: Total distinct queries driving impressions to the page.
    - **top_query_share**: Impression concentration ratio of the primary search query.
- **Features Excluded**: Target leakage indicators (trend_pct), pseudonymous IDs (content_hash_id, client_hash_id), and unaggregated raw timestamps.

## 5. Evaluation

---

- **Split Strategy**: GroupKFold split grouped by client_hash_id (80% train / 20% test) to ensure zero cross-client data leakage and measure out-of-domain client generalization.
- **Model vs Baseline Performance**:

| **Model / Approach**
 | **ROC-AUC** | **PR-AUC** | **Precision@100** | **Lift over Base Rate** |
| --- | --- | --- | --- | --- |
| **Random / Base Rate** | 0.500 | 0.184 | 18.4% | 1.0x |
| **Position-CTR Rule (Baseline)** | 0.651 | 0.382 | 52.0% | 2.8x |
| **LightGBM Refresh Model** | 0.814 | 0.597 | 78.0% | 4.2x |
- **Error Analysis**: Most of the false positives are the top results (with deep impressions) that are the result of broad and information-driven queries. The low CTR is because of this. A false negative means a niche and lengthy tail page with low impressions, still high conversion intent.

![Content Refresh Target Identification: Impressions vs. CTR](./figures/impressions_vs_ctr.png)

*Figure 1: Log-scale distribution of Total Impressions vs. Overall CTR (%) surfacing high-impression, low-CTR candidate pages.*

## 6. Interpretation

---

- **Feature Importance**:
    1. **ctr_position_gap** (38.2% gain): Strongest indicator of underperformance relative to SERP rank.
    2. **log_impressions** (24.1% gain): Prioritizes pages with sufficient traffic leverage.
    3. **top_query_share** (18.5% gain): Identifies intent mismatch where a single primary term dominates impressions.
    4. **mean_position** (12.4% gain): Filters for striking-distance rankings (positions 4–15).
- **Surprises & Negative Results**: Found that page-level metadata (title/description character count) only has very little contribution (≤ 1.2% gain). It means that search intent alignment dominates simple character length heuristics.

## **7. Recommendation**

---

- **Ranked Action Playbook for FlyRank Editors:**
    1. **Tier 1 (Score > 0.85; Reason: High Impression Gap)**: Rewrite Title Tags and H1s to better align with high-impression query intent.
    2. **Tier 2 (Score 0.65–0.85; Reason: Striking Distance Decay)**: Update body copy, re-verify factual freshness, and expand internal linking to push position 6–12 rankings into top 3.
    3. **Tier 3 (Score < 0.65)**: Monitor; no immediate editorial intervention required.
- **Confidence & Limits**: Work provides directional decision support for prioritizing editorial queue allocation. It does not guarantee causal ranking improvements or predict search engine algorithmic shifts.

## 8. Reproducibility

---

* **Commands:**

```bash
git clone [https://github.com/Nirvik-49/Week-1-FlyRank-AI-Assignment.git](https://github.com/Nirvik-49/Week-1-FlyRank-AI-Assignment.git)
cd Week-1-FlyRank-AI-Assignment
pip install -r requirements.txt
python work/build_features.py --seed 42
python work/train_eval.py --seed 42
```

* **Environment & Random Seed:** Python 3.10+, DuckDB 1.0.0+, LightGBM 4.0+, Scikit-Learn 1.3+. Fixed global random seed: 42.
* **Holdout Validation:** `work/holdout_metrics.json` and `work/sealed_frame.parquet` are committed in the repository, containing verified blind evaluation metrics.

## 9. Acknowledgments & data credit

---

Built on the FlyRank ML Internship dataset linking to [https://flyrank.ai](https://flyrank.ai).
