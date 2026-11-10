# Customer Segmentation Project

## What this project does
Segments customers using K-Means clustering and analyses customer purchase patterns.

## Important data note
`data/customer_data.csv` is synthetic demonstration data, not actual company/customer data. Use only approved, anonymised real data for a real business analysis.

## Run the project
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python src/customer_segmentation.py`
5. Review the generated CSV files and charts in `outputs/`.

## Main files
- `data/customer_data.csv` — sample customer dataset
- `src/customer_segmentation.py` — complete analysis script
- `outputs/customer_segments.csv` — customer-level cluster labels
- `outputs/segment_profile.csv` — segment averages
- `outputs/*.png` — charts
- `outputs/Project_Report.md` — internship report

## Features used for clustering
Age, annual income, spending score, purchase frequency, average order value, days since last purchase, monthly website visits and discount usage rate.
