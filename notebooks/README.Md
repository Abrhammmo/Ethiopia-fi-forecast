# 📓 Jupyter Notebooks

This directory contains Jupyter notebooks used for the Ethiopia Financial Inclusion Analysis project. Each notebook focuses on a specific analytical task.

---

## 📁 Notebook Overview

| Notebook                                | Task     | Description                              |
| --------------------------------------- | -------- | ---------------------------------------- |
| `Data Exploration and Enrichment.ipynb` | Task 1-2 | Initial data exploration and enrichment  |
| `Event Impact Modeling.ipynb`           | Task 3   | Event study analysis and impact modeling |
| `Exploratory Data Analysis.ipynb`       | Task 2   | Comprehensive EDA with visualizations    |
| `Forecasting Access and Usage.ipynb`    | Task 4   | Time-series forecasting and scenarios    |

---

## 📓 Notebook Descriptions

### 1. Data Exploration and Enrichment.ipynb

**Associated Task**: Task 1 & Task 2  
**Purpose**: Initial data loading, exploration, and enrichment workflow

**Contents**:

- Data loading and validation
- Initial data quality assessment
- Data enrichment pipeline
- Reference code mapping
- Enrichment logging

**Key Outputs**:

- Enriched dataset ready for analysis
- Documentation of enrichment steps
- Data quality report

---

### 2. Event Impact Modeling.ipynb

**Associated Task**: Task 3  
**Purpose**: Analyze the impact of key events on financial inclusion indicators

**Contents**:

- Event identification and definition
- Event study methodology
- Impact quantification
- Lag structure analysis
- Association matrix generation

**Key Outputs**:

- Event impact coefficients
- Lagged effect estimates
- `event_indicator_association_matrix.csv`
- `impact_links_enriched.csv`

---

### 3. Exploratory Data Analysis.ipynb

**Associated Task**: Task 2  
**Purpose**: Comprehensive exploratory data analysis of financial inclusion metrics

**Contents**:

- Account ownership trends analysis
- Usage patterns examination
- Infrastructure correlation analysis
- Gender gap assessment
- Channel comparison (digital vs traditional)
- Trend visualization

**Key Outputs**:

- Statistical insights and findings
- Data visualizations (saved to `reports/figures/`)
- Correlation analysis results

---

### 4. Forecasting Access and Usage.ipynb

**Associated Task**: Task 4  
**Purpose**: Develop and evaluate forecasting models for financial inclusion

**Contents**:

- Time-series model development
- Linear trend models
- Log-linear growth models
- Event-augmented forecasting
- Scenario generation (Optimistic, Base, Pessimistic)
- Model evaluation and validation

**Key Outputs**:

- Forecast data (`all_forecasts.csv`)
- Scenario projections
- Model performance metrics
- Confidence intervals

---

## 🚀 Running the Notebooks

### Option 1: Using Jupyter Notebook

```bash
# From project root
jupyter notebook notebooks/
```

### Option 2: Using JupyterLab

```bash
# From project root
jupyter lab notebooks/
```

### Option 3: Using VS Code

1. Open the notebook file in VS Code
2. Ensure Jupyter extension is installed
3. Select kernel and run cells

---

## 📋 Notebook Structure Best Practices

Each notebook should follow this structure:

1. **Title & Description** - Clear title and purpose statement
2. **Imports** - All required libraries
3. **Configuration** - Settings and constants
4. **Data Loading** - Functions to load data
5. **Analysis** - Main analytical code
6. **Visualization** - Plots and charts
7. **Conclusions** - Summary of findings
8. **Export** - Save results if applicable

---

## 📁 Related Files

- **Data**: `data/processed/` and `data/raw/`
- **Reports**: `reports/figures/` (exported visualizations)
- **Models**: `models/` (trained forecasting models)
- **Source Code**: `src/` (reusable Python modules)

---

## 📝 Notes

- Notebooks are numbered by task for easy navigation
- All notebooks use absolute paths from project root
- Data is loaded using functions from `src/` modules
- Visualizations are saved to `reports/figures/`
- Restart kernel and run all cells before sharing

---

_Last Updated: February 2025_
