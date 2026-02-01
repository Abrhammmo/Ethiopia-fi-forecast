# Event Impact Modeling Methodology

## Ethiopia Financial Inclusion Forecast

**Date:** February 2025  
**Author:** Ethiopia FI Forecast Team

---

## 1. Executive Summary

This document describes the methodology used to model how events (policies, product launches, infrastructure investments) affect financial inclusion indicators in Ethiopia. The model translates documented impact links into a predictive framework that can estimate how indicators change when events occur.

**Key Deliverables:**

- Event-Indicator Association Matrix
- Temporal impact model with lag effects
- Validation against historical data (Telebirr case study)
- Confidence assessment for all estimates

---

## 2. Conceptual Framework

### 2.1 Events and Their Types

Events are discrete occurrences that can influence financial inclusion outcomes:

| Event Type         | Description                                        | Examples                   |
| ------------------ | -------------------------------------------------- | -------------------------- |
| **Product Launch** | Introduction of new financial products or services | Telebirr, M-Pesa           |
| **Market Entry**   | New competitors entering the market                | Safaricom Ethiopia         |
| **Policy Change**  | Government regulatory or policy changes            | FX Liberalization, NFIS-II |
| **Infrastructure** | Development of foundational systems                | Fayda Digital ID, EthioPay |
| **Pricing Change** | Significant changes to pricing                     | Safaricom Price Hike       |

### 2.2 Impact Mechanism Types

Each event-indicator relationship is classified by mechanism:

| Mechanism    | Description                         | Example                              |
| ------------ | ----------------------------------- | ------------------------------------ |
| **Direct**   | Immediate, effect                   | Teleb observableirr → Telebirr Users |
| **Indirect** | Effect through intermediate factors | Safaricom → Data Affordability       |
| **Enabling** | Removes barriers for other effects  | Fayda ID → Account Ownership         |

---

## 3. Functional Forms

### 3.1 Temporal Impact Function

The core model uses a linear ramp function with lag:

```
E(t) = {
    0                                          if t < lag
    total_effect × (t - lag) / ramp           if lag ≤ t < lag + ramp
    total_effect                              if t ≥ lag + ramp
}
```

Where:

- **t**: Months since event occurrence
- **total_effect**: Maximum impact magnitude (percentage points)
- **lag**: Delay before effect starts (months)
- **ramp**: Time to reach full effect (months)

### 3.2 Ramp Period by Relationship Type

| Relationship Type | Default Ramp | Rationale                             |
| ----------------- | ------------ | ------------------------------------- |
| Direct            | 6 months     | Direct subscriber acquisition is fast |
| Indirect          | 12 months    | Market effects take time to propagate |
| Enabling          | 24 months    | Infrastructure enables other effects  |

### 3.3 Lag Period Selection

Lag periods are derived from:

- **Empirical evidence**: Observed time to effect in Ethiopia
- **Comparable country data**: Evidence from Kenya, India, Tanzania
- **Theoretical estimation**: Logical deduction for new events

---

## 4. Event-Indicator Association Matrix

### 4.1 Matrix Structure

The association matrix is structured as:

|                      | ACC_OWNERSHIP | ACC_MM_ACCOUNT | USG_P2P_COUNT | ... |
| -------------------- | :-----------: | :------------: | :-----------: | :-: |
| **Telebirr Launch**  |     +15.0     |      +5.0      |     +25.0     | ... |
| **Safaricom Entry**  |       -       |     +15.0      |       -       | ... |
| **M-Pesa Launch**    |     +10.0     |      +5.0      |       -       | ... |
| **Fayda Digital ID** |     +10.0     |       -        |       -       | ... |

### 4.2 Indicator Codes and Definitions

| Code                | Indicator                 | Direction     | Description                          |
| ------------------- | ------------------------- | ------------- | ------------------------------------ |
| ACC_OWNERSHIP       | Account Ownership Rate    | Higher better | % of adults with financial account   |
| ACC_MM_ACCOUNT      | Mobile Money Account Rate | Higher better | % with mobile money account          |
| ACC_4G_COV          | 4G Population Coverage    | Higher better | % covered by 4G network              |
| USG_TELEBIRR_USERS  | Telebirr Registered Users | Higher better | Count of registered users            |
| USG_MPESA_USERS     | M-Pesa Registered Users   | Higher better | Count of registered users            |
| USG_P2P_COUNT       | P2P Transaction Count     | Higher better | Monthly P2P transactions             |
| USG_DIGITAL_PAYMENT | Digital Payment Usage     | Higher better | % using digital payments             |
| AFF_DATA_INCOME     | Data Affordability Index  | Lower better  | % of GNI for 1GB data                |
| GEN_GAP_ACC         | Gender Gap                | Lower better  | Male-female gap in account ownership |

---

## 5. Impact Estimates by Event

### 5.1 Telebirr Launch (EVT_0001)

**Date:** May 17, 2021  
**Type:** Product Launch

| Indicator          | Impact (pp) | Lag | Confidence | Evidence                  |
| ------------------ | ----------- | --- | ---------- | ------------------------- |
| ACC_OWNERSHIP      | +15.0       | 12  | Medium     | Kenya M-Pesa analogy      |
| ACC_MM_ACCOUNT     | +5.0        | 6   | HIGH       | Ethiopian data validation |
| USG_TELEBIRR_USERS | -           | 3   | High       | Ethio Telecom reports     |
| USG_P2P_COUNT      | +25.0       | 6   | Medium     | New payment channel       |

**Validation:** Mobile money account rate: 4.7% (2021) → 9.45% (2024) = +4.75pp
Model estimate: +5.0pp → Relative error: 5.3% ✓ VALIDATED

### 5.2 Safaricom Ethiopia Launch (EVT_0002)

**Date:** August 1, 2022  
**Type:** Market Entry

| Indicator       | Impact (pp) | Lag | Confidence | Evidence                 |
| --------------- | ----------- | --- | ---------- | ------------------------ |
| ACC_4G_COV      | +15.0       | 12  | Medium     | Network investment       |
| AFF_DATA_INCOME | -20.0       | 12  | Medium     | Rwanda competition study |

**Note:** Competition effects typically reduce prices by 15-25%

### 5.3 M-Pesa Ethiopia Launch (EVT_0003)

**Date:** August 1, 2023  
**Type:** Product Launch

| Indicator       | Impact (pp) | Lag | Confidence | Evidence               |
| --------------- | ----------- | --- | ---------- | ---------------------- |
| USG_MPESA_USERS | -           | 3   | High       | Subscriber acquisition |
| ACC_MM_ACCOUNT  | +5.0        | 6   | Medium     | Second provider effect |

### 5.4 Fayda Digital ID Program (EVT_0004)

**Date:** January 1, 2024  
**Type:** Infrastructure

| Indicator     | Impact (pp) | Lag | Confidence | Evidence                     |
| ------------- | ----------- | --- | ---------- | ---------------------------- |
| ACC_OWNERSHIP | +8.0        | 24  | Medium     | India Aadhaar (conservative) |
| GEN_GAP_ACC   | -5.0        | 24  | Medium     | India Aadhaar gender impact  |

**Note:** India Aadhaar showed 15-20pp impact, but Ethiopia has lower baseline

### 5.5 Government Wage Digitization (EVT_0011)

**Date:** January 1, 2022  
**Type:** Policy

| Indicator           | Impact (pp) | Lag | Confidence | Evidence                 |
| ------------------- | ----------- | --- | ---------- | ------------------------ |
| USG_DIGITAL_PAYMENT | +7.0        | 9   | Medium     | Ghana/Kenya G2P evidence |

**Mechanism:** Recurring payments create habitual digital payment usage

---

## 6. Comparable Country Evidence

### 6.1 Kenya (M-Pesa Reference)

- **Impact on Account Ownership:** +20pp over 5 years (2007-2012)
- **Source:** Global Findex, World Bank
- **Application:** Telebirr effect on ACC_OWNERSHIP

### 6.2 India (Aadhaar Reference)

- **Impact on Account Opening:** +15-20pp following Aadhaar rollout
- **Source:** World Bank research, NITI Aayog
- **Application:** Fayda Digital ID effect on ACC_OWNERSHIP

### 6.3 Rwanda (Competition Effects)

- **Impact on Data Prices:** -20-30% following market liberalization
- **Source:** GSMA Mobile Connectivity Index
- **Application:** Safaricom entry effect on AFF_DATA_INCOME

### 6.4 Tanzania (Interoperability)

- **Impact on Mobile Money Usage:** +20% after interoperability
- **Source:** Bill & Melinda Gates Foundation research
- **Application:** M-Pesa Interop effect on USG_MPESA_ACTIVE

---

## 7. Model Validation

### 7.1 Telebirr Case Study

**Hypothesis:** Telebirr launch would increase mobile money account ownership

**Observed Data:**

- Pre-Telebirr (2021): 4.7%
- Post-Telebirr (2024): 9.45%
- Actual Change: +4.75pp

**Model Prediction:**

- Estimated Impact: +5.0pp
- Time to Full Effect: 6 months (direct relationship)

**Result:** ✓ VALIDATED

- Relative Error: 5.3% (< 30% threshold)
- Model slightly overestimates (5.0pp vs 4.75pp actual)

**Adjustment:** Updated refined estimate to 4.75pp based on empirical validation

### 7.2 Validation Limitations

- Limited pre/post data for most events
- Cannot isolate event effects from other factors
- Future events cannot be validated until data is available

---

## 8. Key Assumptions

### 8.1 Causal Direction

**Assumption:** Events cause indicator changes, not reverse causation

**Justification:**

- Temporal sequence: Events precede observed changes
- Theoretical basis: Product launches logically precede adoption
- Comparable evidence: Similar patterns in other countries

### 8.2 No Effect Saturation

**Assumption:** Effects continue to compound linearly

**Limitation:** In reality, adoption curves follow S-curve patterns

### 8.3 Independent Effects

**Assumption:** Multiple events add linearly (no interaction effects)

**Limitation:** Real-world effects may amplify or dampen each other

### 8.4 Sustained Effects

**Assumption:** Once achieved, effects persist

**Limitation:** Some effects may decay over time (e.g., if service quality declines)

### 8.5 Homogeneous Population

**Assumption:** Effects apply uniformly across population subgroups

**Limitation:** Gender, income, and geographic differences likely exist

---

## 9. Limitations and Uncertainties

### 9.1 Data Limitations

| Limitation                      | Impact                                   | Mitigation                             |
| ------------------------------- | ---------------------------------------- | -------------------------------------- |
| Limited Ethiopian pre/post data | Most estimates from comparable countries | Used conservative estimates            |
| Short time series               | Cannot observe long-term effects         | Extrapolated from comparable countries |
| Future events                   | Cannot validate projections              | Marked as projections                  |

### 9.2 Methodological Limitations

| Limitation             | Impact                       | Mitigation                              |
| ---------------------- | ---------------------------- | --------------------------------------- |
| Linear addition        | May not capture interactions | Noted in documentation                  |
| No counterfactual      | Cannot prove causality       | Used comparable country evidence        |
| Single functional form | May not fit all impacts      | Default parameters by relationship type |

### 9.3 Confidence Assessment

| Confidence Level | Definition            | Criteria                                    |
| ---------------- | --------------------- | ------------------------------------------- |
| **HIGH**         | Empirically validated | Ethiopian data confirms estimate            |
| **MEDIUM**       | Good evidence         | Comparable country data + theoretical basis |
| **LOW**          | Theoretical estimate  | Limited evidence available                  |

---

## 10. Sensitivity Analysis

### 10.1 Key Sensitivities

1. **Ramp Period Sensitivity:**
   - 6-month vs 12-month ramp changes timing but not total effect
2. **Comparable Country Adjustments:**
   - Kenya factor: 1.0 (similar trajectory)
   - India factor: 0.7 (different context)
   - Rwanda factor: 0.8 (similar development stage)

3. **Confidence Weighting:**
   - High confidence (1.0) vs Low confidence (0.3) affects aggregate projections

---

## 11. Future Improvements

### 11.1 Data Collection Priorities

1. Collect more pre/post data for 2024-2025 events
2. Disaggregate by gender, income, urban/rural
3. Track sentiment and satisfaction alongside adoption

### 11.2 Methodological Improvements

1. Incorporate interaction effects between events
2. Develop S-curve adoption models for major events
3. Implement Bayesian updating as new data arrives

### 11.3 Model Extensions

1. Sub-population disaggregation
2. Regional variation modeling
3. Scenario analysis (optimistic vs conservative)

---

## 12. References

1. World Bank Global Findex (2014, 2017, 2021, 2024)
2. GSMA Mobile Connectivity Index
3. Ethio Telecom LEAD Reports
4. EthSwitch Annual Reports
5. NBE Financial Inclusion Reports
6. Safaricom Ethiopia Quarterly Results
7. A4AI/ITU Affordability Reports

---

## Appendix A: Event Key

| ID       | Event Name         | Date       | Type           |
| -------- | ------------------ | ---------- | -------------- |
| EVT_0001 | Telebirr Launch    | 2021-05-17 | Product Launch |
| EVT_0002 | Safaricom Ethiopia | 2022-08-01 | Market Entry   |
| EVT_0003 | M-Pesa Ethiopia    | 2023-08-01 | Product Launch |
| EVT_0004 | Fayda Digital ID   | 2024-01-01 | Infrastructure |
| EVT_0005 | FX Reform          | 2024-07-29 | Policy         |
| EVT_0009 | NFIS-II Strategy   | 2021-09-01 | Policy         |
| EVT_0011 | Wage Digitization  | 2022-01-01 | Policy         |

---

_Document generated: February 2025_
_Version: 1.0.0_
