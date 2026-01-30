# Conclusions: Data Exploration and Enrichment for Ethiopia Financial Inclusion Dataset

## Executive Summary

This analysis covers Task-1 of the Ethiopia Financial Inclusion Forecast project, focusing on data exploration and enrichment of the unified dataset. The work establishes a robust foundation for subsequent analytical tasks including impact estimation, forecasting, and dashboard development.

---

## 1. Dataset Overview

| Metric              | Value                    |
| ------------------- | ------------------------ |
| Total Records       | 43                       |
| Observation Records | 30                       |
| Event Records       | 10                       |
| Target Records      | 3                        |
| Temporal Range      | 2014-12-31 to 2030-12-31 |

The dataset follows a unified schema design where all records share a common structure, with `record_type` determining interpretation. Events are pillar-agnostic by design, and causal assumptions are expressed only via `impact_link` records.

---

## 2. Pillar Distribution Analysis

### 2.1 Observation Distribution by Pillar

| Pillar        | Count | Percentage |
| ------------- | ----- | ---------- |
| ACCESS        | 14    | 46.7%      |
| USAGE         | 11    | 36.7%      |
| GENDER        | 4     | 13.3%      |
| AFFORDABILITY | 1     | 3.3%       |

**Key Insights:**

- **ACCESS** dominates with strong coverage of mobile account ownership, 4G coverage, and digital ID (Fayda)
- **USAGE** shows robust P2P transaction and mobile money activity metrics
- **GENDER** indicators reveal gaps in women's financial access
- **AFFORDABILITY** is significantly underrepresented with only 1 observation

### 2.2 Event Distribution by Category

| Category       | Count |
| -------------- | ----- |
| Product Launch | 2     |
| Policy         | 2     |
| Infrastructure | 2     |
| Market Entry   | 1     |
| Partnership    | 1     |
| Milestone      | 1     |
| Pricing        | 1     |

---

## 3. Event Inventory Analysis

The dataset captures 10 pivotal events in Ethiopia's digital finance evolution (2021-2025):

| Event               | Date     | Category       | Impact Area              |
| ------------------- | -------- | -------------- | ------------------------ |
| Telebirr Launch     | May 2021 | product_launch | Mobile money adoption    |
| NFIS-II Strategy    | Sep 2021 | policy         | National FI strategy     |
| Safaricom Ethiopia  | Aug 2022 | market_entry   | Competition/innovation   |
| M-Pesa Ethiopia     | Aug 2023 | product_launch | Competition/usage growth |
| Fayda Digital ID    | Jan 2024 | infrastructure | Identity/access enabler  |
| FX Liberalization   | Jul 2024 | policy         | Market access            |
| P2P > ATM Milestone | Oct 2024 | milestone      | Usage shift              |
| G2P Expansion       | 2023     | policy         | Recurring usage driver   |
| EthioPay IPS        | Dec 2025 | infrastructure | Real-time payments       |
| Price Adjustment    | Dec 2025 | pricing        | Affordability impact     |

---

## 4. Impact Link Framework Analysis

### 4.1 Impact Distribution

| Metric                 | Value   |
| ---------------------- | ------- |
| Total Impact Links     | 14      |
| Literature-based       | 7 (50%) |
| Empirical (comparable) | 6 (43%) |
| Theoretical            | 1 (7%)  |

### 4.2 Lag Structure

| Statistic  | Value      |
| ---------- | ---------- |
| Mean Lag   | 8.4 months |
| Median Lag | 6 months   |
| Min Lag    | 1 month    |
| Max Lag    | 24 months  |

**Interpretation:** The 6-month median lag suggests most interventions have medium-term effects on indicators, supporting intervention-aware modeling approaches.

### 4.3 Impact Magnitude

| Magnitude | Count | Description                 |
| --------- | ----- | --------------------------- |
| Medium    | 8     | Moderate, measurable change |
| High      | 5     | Significant transformation  |
| Low       | 1     | Marginal impact             |

### 4.4 Impact Distribution by Pillar

| Pillar        | Count | Focus Area           |
| ------------- | ----- | -------------------- |
| USAGE         | 6     | Transaction behavior |
| ACCESS        | 4     | Account ownership    |
| AFFORDABILITY | 3     | Cost barriers        |
| GENDER        | 1     | Gender gaps          |

---

## 5. Data Enrichment Outcomes

### 5.1 New Observations Added

1. **ACC_4G_COV (55%)**
   - Description: 4G population coverage in Ethiopia
   - Date: 2023-12-31
   - Source: GSMA Mobile Connectivity Index
   - Purpose: Infrastructure proxy for digital payment adoption

2. **USG_G2P_DIGITIZED (18%)**
   - Description: Share of government-to-person payments digitized
   - Date: 2024-12-31
   - Source: National Bank of Ethiopia
   - Purpose: Track policy-driven digital payment adoption

### 5.2 New Events Added

1. **Government Wage Digitization Initiative**
   - Date: 2022-01-01
   - Category: policy
   - Expected Impact: Increase DFS usage via recurring payments

2. **Expansion of Digital G2P Payments (PSNP & Pensions)**
   - Date: 2023-01-01
   - Category: policy
   - Purpose: Recurring government payments create sustained digital account usage

### 5.3 New Impact Links Added

1. **Wage Digitization → Digital Payments**
   - Evidence: Literature (Ghana)
   - Lag: 6 months
   - Magnitude: medium

2. **G2P Digitization → Digital Payment Usage**
   - Evidence: Literature (Kenya)
   - Lag: 9 months
   - Magnitude: medium
   - Estimate: 5-10pp increase

---

## 6. Reference Code System

Three new reference codes added to maintain governance:

| Code                        | Type        | Purpose                                      |
| --------------------------- | ----------- | -------------------------------------------- |
| `USG_G2P_DIGITIZED`         | indicator   | Tracks digital G2P payment adoption          |
| `EVT_G2P_DIGITAL_EXPANSION` | event       | Policy-driven payment digitization           |
| `IMP_G2P_USAGE_EFFECT`      | impact_link | Causal relationship between policy and usage |

---

## 7. Data Quality Assessment

### 7.1 Strengths

- Well-structured unified schema
- Clear separation of observations, events, and causal links
- Comparable country evidence for impact estimation
- Comprehensive event inventory covering key policy milestones

### 7.2 Limitations

- Sparse time series for many indicators
- Single-point observations require careful modeling
- Impact magnitudes are directional, not precise elasticities
- Lags estimated from literature, not observed
- Gender and affordability pillars underrepresented
- Several metadata fields have significant null rates

### 7.3 Data Integrity

- All limitations explicitly preserved for policy-grade analytical standards
- No smoothing or imputation applied to preserve data authenticity
- Duplicate protection implemented for reference codes

---

## 8. Strategic Implications

### 8.1 Key Findings

1. **G2P Digitization is Critical**
   - Addition of G2P indicators reflects its role as a key adoption driver
   - Literature evidence (Kenya, Ghana) supports this focus

2. **Infrastructure Gap Remains**
   - 55% 4G coverage indicates connectivity barriers
   - Infrastructure investment needed to support digital payments growth

3. **Policy Impact Timing**
   - 6-9 month average lag supports intervention-aware modeling
   - Patience required before measuring policy impact

4. **Gender Focus Needed**
   - Only 5 records on gender gaps
   - Opportunity for targeted data collection

5. **Affordability Underrepresented**
   - Single observation suggests research gap
   - Price changes (Dec 2025) may affect affordability

### 8.2 Recommendations

1. Prioritize G2P digitization monitoring as a leading indicator
2. Expand gender-disaggregated data collection
3. Develop affordability metrics for price sensitivity analysis
4. Use comparable country evidence for gap-filling where appropriate
5. Maintain intervention-aware modeling approaches

---

## 9. Enriched Output Files

The following files were generated from this enrichment phase:

| File                                  | Location        | Description                                |
| ------------------------------------- | --------------- | ------------------------------------------ |
| ethiopia_fi_unified_data_enriched.csv | data/processed/ | Main dataset with new observations/events  |
| impact_links_enriched.csv             | data/processed/ | Impact links with new causal relationships |
| updated_reference_codes.csv           | data/processed/ | Reference codes with new entries           |

