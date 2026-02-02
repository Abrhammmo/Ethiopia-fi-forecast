# 📊 Ethiopia Financial Inclusion Analysis & Forecasting

A comprehensive financial inclusion analytics initiative focused on **Ethiopia**, covering data enrichment, exploratory analysis, impact modeling, forecasting, and interactive dashboard development.

---

## 📋 Project Overview

This project delivers a complete analytical pipeline for understanding and forecasting financial inclusion in Ethiopia, from data preparation to actionable insights and stakeholder communication.

### Tasks Completed

| Task   | Status      | Description                                 |
| ------ | ----------- | ------------------------------------------- |
| Task 1 | ✅ Complete | Data Enrichment & Reference Framework       |
| Task 2 | ✅ Complete | Exploratory Data Analysis (EDA)             |
| Task 3 | ✅ Complete | Event Impact Modeling & Feature Engineering |
| Task 4 | ✅ Complete | Time-Series Forecasting & Scenario Analysis |
| Task 5 | ✅ Complete | Dashboard Development & Business Insights   |

---

## ✅ Task 1: Data Enrichment & Reference Framework

### 🎯 Objective

To enrich the raw financial inclusion dataset by standardizing indicators, introducing contextual observations/events, and creating a transparent enrichment log.

### 🔧 Key Activities

1. **Reference Code Management**
   - Extended `reference_codes.csv` with new observations, events, and impact links
   - Ensured consistency in code naming, descriptions, and category types

2. **Data Record Enrichment**
   - Added new records mapping indicators to reference codes
   - Linked time-specific changes to real-world events
   - Maintained temporal alignment without artificial data leakage

3. **Enrichment Logging**
   - Produced comprehensive documentation of all additions and modifications
   - Ensured full traceability for audit and review purposes

### 📁 Task-1 Outputs

| File                                                   | Description                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| `data/reference_codes.csv`                             | Centralized taxonomy for indicators, events, and observations |
| `data/processed/ethiopia_fi_unified_data_enriched.csv` | Enriched analytical dataset                                   |
| `reports/data_enrichment_log.md`                       | Detailed enrichment documentation                             |

---

## ✅ Task 2: Exploratory Data Analysis (EDA)

### 🎯 Objective

To explore and explain patterns in financial access and usage in Ethiopia, focusing on account ownership, infrastructure drivers, and distributional gaps.

### 📊 Key Findings

1. **Slow Ownership Growth Despite Massive Registrations**
   - Account ownership increased only ~+3pp (2021–2024)
   - Registered ≠ active ≠ owned

2. **Usage Growing Faster Than Access**
   - Transaction volumes and values grew dramatically
   - Higher intensity per user rather than broad-based new adoption

3. **Infrastructure Appears to Lead Usage**
   - Rapid 4G expansion preceded transaction growth
   - Connectivity is a leading indicator of digital financial usage

4. **Persistent Gender Gap**
   - Female ownership remains materially lower
   - Structural barriers persist (device access, ID)

### 📁 Task-2 Outputs

| File                                              | Description                 |
| ------------------------------------------------- | --------------------------- |
| `notebooks/Data Exploration and Enrichment.ipynb` | Main EDA notebook           |
| `notebooks/Exploratory Data Analysis.ipynb`       | Additional analysis         |
| `reports/figures/`                                | Exported EDA visualizations |

---

## ✅ Task 3: Event Impact Modeling & Feature Engineering

### 🎯 Objective

To quantify the impact of key policy and infrastructure events on financial inclusion indicators using event study methodology.

### 🔧 Key Activities

1. **Event Identification**
   - Identified key events: Telebirr launch, Safaricom entry, M-Pesa launch, FX reforms
   - Defined event windows and control periods

2. **Impact Analysis**
   - Applied event study methodology to measure pre/post event changes
   - Estimated lag structures for event impact realization

3. **Feature Engineering**
   - Created event indicator variables
   - Engineered lagged features for temporal relationships
   - Built association matrix between events and indicators

### 📁 Task-3 Outputs

| File                                                    | Description                   |
| ------------------------------------------------------- | ----------------------------- |
| `notebooks/Event Impact Modeling.ipynb`                 | Impact analysis methodology   |
| `data/processed/event_indicator_association_matrix.csv` | Event-indicator relationships |
| `data/processed/impact_links_enriched.csv`              | Impact link definitions       |
| `reports/impact_modeling_methodology.md`                | Methodology documentation     |

---

## ✅ Task 4: Time-Series Forecasting & Scenario Analysis

### 🎯 Objective

To develop forecasting models for financial inclusion indicators and generate scenario-based projections.

### 🔧 Key Activities

1. **Model Development**
   - Implemented linear trend models
   - Built log-linear growth models
   - Created event-augmented forecasting models

2. **Scenario Analysis**
   - Developed Optimistic, Base, and Pessimistic scenarios
   - Incorporated policy assumptions into projections
   - Generated confidence intervals for forecasts

3. **Model Evaluation**
   - Evaluated forecast accuracy using multiple metrics
   - Compared model performance across scenarios
   - Validated projections against historical data

### 📁 Task-4 Outputs

| File                                           | Description                  |
| ---------------------------------------------- | ---------------------------- |
| `notebooks/Forecasting Access and Usage.ipynb` | Forecasting methodology      |
| `data/processed/all_forecasts.csv`             | Forecast data with scenarios |
| `data/processed/ACC_OWNERSHIP_scenarios.csv`   | Account ownership scenarios  |
| `data/processed/USG_P2P_COUNT_scenarios.csv`   | P2P transaction scenarios    |
| `models/`                                      | Trained forecast models      |

---

## ✅ Task 5: Dashboard Development & Business Insights

### 🎯 Objective

To create an interactive dashboard enabling stakeholders to explore data, understand event impacts, and view forecasts.

### 📊 Dashboard Sections

#### 1. Overview Page

- Key metrics summary cards (account ownership, P2P transactions, P2P/ATM ratio)
- Progress gauge toward 60% financial inclusion target
- Growth rate highlights and key insights

#### 2. Trends Page

- Interactive time series plots (line, bar, area charts)
- Date range selector for filtering
- Channel comparison view with normalized trends
- Period-over-period growth rates and CAGR analysis

#### 3. Forecasts Page

- Forecast visualizations with confidence intervals
- Scenario selection (Optimistic, Base, Pessimistic)
- Model selection options (Linear, Log-Linear, Event-Augmented, ARIMA)
- Key projected milestones table
- Forecast uncertainty analysis

#### 4. Inclusion Projections Page

- Financial inclusion rate projections toward 60% target
- Scenario selector with detailed descriptions
- Years-to-target estimation
- Milestone tracker by scenario
- Policy recommendations

### 🎯 Key Metrics Displayed

| Metric             | Description                                |
| ------------------ | ------------------------------------------ |
| Account Ownership  | % of adults with financial accounts        |
| P2P Transactions   | Peer-to-peer transaction volume (millions) |
| P2P/ATM Ratio      | Digital vs traditional channel usage       |
| Mobile Money Users | Number of active users (millions)          |

---

## 🛠️ Installation & Running

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git for version control

### Installation

1. **Clone the repository and navigate to the project directory:**

```bash
git clone <repository-url>
cd Ethiopia-fi-forecast
```

2. **Create a virtual environment (recommended):**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ethiopia-fi python=3.10
conda activate ethiopia-fi
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Dashboard

To start the interactive dashboard, run:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

### Running Notebooks

To explore the analysis notebooks:

```bash
jupyter notebook notebooks/
```

---

## 📁 Project Structure

```
Ethiopia-fi-forecast/
├── dashboard/
│   └── app.py              # Streamlit dashboard application
├── data/
│   ├── processed/          # Processed and enriched datasets
│   └── raw/                # Raw data files
├── models/                 # Trained forecast models and metadata
├── notebooks/              # Jupyter notebooks for analysis
├── reports/                # Analysis reports and documentation
│   └── figures/            # Exported visualizations
├── src/                    # Source code modules
│   └── forecast/           # Forecasting utilities
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📈 Key Insights & Recommendations

### Progress Toward 60% Target

| Scenario    | 50% Inclusion | 60% Target |
| ----------- | ------------- | ---------- |
| Optimistic  | Q1 2026       | Q1 2028    |
| Base Case   | Q3 2026       | Q3 2029    |
| Pessimistic | Q1 2027       | Q4 2031    |

### Policy Recommendations

| Priority | Recommendation                      | Expected Impact        |
| -------- | ----------------------------------- | ---------------------- |
| High     | Scale G2P digitization              | +3-5pp active usage    |
| High     | Expand agent banking in rural areas | +2-4pp access          |
| Medium   | Financial literacy programs         | +1-2pp usage intensity |
| Medium   | Address gender barriers             | -2-3pp gender gap      |

---

## 🔧 Technical Details

### Dependencies

Key packages used in this project:

- **Data Processing**: pandas, numpy, python-dateutil
- **Visualization**: matplotlib, seaborn, plotly
- **Forecasting**: scipy, scikit-learn, statsmodels
- **Dashboard**: streamlit
- **Notebooks**: jupyter, ipykernel

### Data Sources

- Global Findex Database (World Bank)
- National Bank of Ethiopia
- Ethiopian Communications Authority
- Telecom operator reports

---

## 📝 Notes

- The dashboard automatically falls back to sample data if processed data files are not available
- All data is cached for performance optimization
- Charts are interactive with zoom, pan, and hover details
- Data export functionality available in CSV format

---

## 📧 Contact

For questions or collaboration, please contact the Financial Inclusion Analytics Team.

---

_Last Updated: February 2025_
