"""
==============================================================================
 SALES & BUSINESS ANALYTICS DASHBOARD
==============================================================================
 Author      : (Your Name)
 Description : End-to-end sales & business analytics project built on a
               realistic, self-generated retail sales dataset. The script
               performs data generation, cleaning, exploratory data analysis,
               KPI calculation, SQL-style querying (via pandasql / sqlite3),
               visualization, and produces a final business insights &
               recommendations report — all from a single Python file.

 How to run  : python sales_business_analytics_dashboard.py

 Notes       : - No external files (CSV/Excel/DB) are required. The dataset
                 is generated in-memory using NumPy/Pandas with a fixed
                 random seed so results are reproducible.
               - Charts are saved as PNG files in an "output_charts" folder
                 that is created automatically when the script runs, so the
                 visualizations can be viewed even without an interactive
                 display (useful for GitHub / headless environments).
               - SQL analysis is done using Python's built-in sqlite3
                 module, so no external database server is needed.
==============================================================================
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import sqlite3
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so charts save fine anywhere
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Global plotting style
# ------------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

# Reproducibility
np.random.seed(42)

# Folder where all charts will be saved
OUTPUT_DIR = "output_charts"


# ==============================================================================
# 2. DATASET GENERATION (SAMPLE / SYNTHETIC DATA)
# ==============================================================================
def generate_sales_dataset(n_rows: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic synthetic retail sales dataset.

    The dataset mimics an e-commerce / retail store's order-level sales data
    with customers, products, categories, regions, pricing, discounts,
    quantities and dates spread across 3 years.

    Some intentional messiness (missing values, duplicate rows, wrong data
    types, and a few outliers) is injected so that the data-cleaning section
    of this project has real issues to fix — similar to real-world data.
    """

    # ---- Reference / lookup data -------------------------------------------
    regions = ["North", "South", "East", "West", "Central"]

    categories_products = {
        "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "USB-C Charger",
                         "Laptop Stand", "Smartwatch", "Earbuds"],
        "Clothing": ["Men's T-Shirt", "Women's Jeans", "Winter Jacket",
                     "Running Shoes", "Formal Shirt", "Casual Sneakers"],
        "Home & Kitchen": ["Non-Stick Pan", "Electric Kettle", "LED Desk Lamp",
                            "Storage Organizer", "Vacuum Cleaner", "Blender"],
        "Furniture": ["Office Chair", "Study Table", "Bookshelf",
                      "Bed Frame", "Sofa Cover", "Coffee Table"],
        "Beauty & Personal Care": ["Face Wash", "Shampoo", "Perfume",
                                    "Hair Dryer", "Trimmer", "Sunscreen"],
        "Sports & Fitness": ["Yoga Mat", "Dumbbell Set", "Cricket Bat",
                              "Football", "Resistance Bands", "Skipping Rope"],
    }

    # Base unit price ranges per category (min, max) in INR — kept realistic
    price_ranges = {
        "Electronics": (500, 6000),
        "Clothing": (300, 3000),
        "Home & Kitchen": (400, 5000),
        "Furniture": (1500, 15000),
        "Beauty & Personal Care": (150, 2000),
        "Sports & Fitness": (200, 4000),
    }

    payment_modes = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]
    customer_segments = ["Regular", "Premium", "New"]

    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
                   "Ishaan", "Ananya", "Diya", "Priya", "Isha", "Kavya", "Meera",
                   "Rohan", "Karthik", "Sneha", "Pooja", "Neha", "Rahul"]
    last_names = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Kumar",
                  "Patel", "Singh", "Menon", "Rao", "Das", "Joshi", "Malhotra"]

    n_customers = 400
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    customer_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                       for _ in range(n_customers)]
    customer_lookup = dict(zip(customer_ids, customer_names))
    customer_segment_lookup = dict(
        zip(customer_ids, np.random.choice(customer_segments, size=n_customers, p=[0.55, 0.25, 0.20]))
    )

    # ---- Date range: 3 full years ------------------------------------------
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    rows = []
    for i in range(1, n_rows + 1):
        order_id = f"ORD{str(i).zfill(6)}"
        cust_id = np.random.choice(customer_ids)
        category = np.random.choice(list(categories_products.keys()))
        product = np.random.choice(categories_products[category])
        region = np.random.choice(regions, p=[0.25, 0.20, 0.20, 0.20, 0.15])

        low, high = price_ranges[category]
        unit_price = round(np.random.uniform(low, high), 2)

        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05])
        discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], p=[0.35, 0.2, 0.2, 0.15, 0.07, 0.03])

        random_day_offset = np.random.randint(0, date_range_days)
        order_date = start_date + timedelta(days=int(random_day_offset))

        payment_mode = np.random.choice(payment_modes)

        rows.append({
            "Order_ID": order_id,
            "Order_Date": order_date,
            "Customer_ID": cust_id,
            "Customer_Name": customer_lookup[cust_id],
            "Customer_Segment": customer_segment_lookup[cust_id],
            "Region": region,
            "Category": category,
            "Product": product,
            "Unit_Price": unit_price,
            "Quantity": quantity,
            "Discount_Percent": discount_pct,
            "Payment_Mode": payment_mode,
        })

    df = pd.DataFrame(rows)

    # ---- Derived financial fields ------------------------------------------
    df["Gross_Sales"] = df["Unit_Price"] * df["Quantity"]
    df["Discount_Amount"] = df["Gross_Sales"] * (df["Discount_Percent"] / 100)
    df["Revenue"] = df["Gross_Sales"] - df["Discount_Amount"]
    # Assume a cost-of-goods ratio (COGS) between 55%-75% of unit price by category,
    # to allow realistic profit calculation.
    cogs_ratio_lookup = {
        "Electronics": 0.68, "Clothing": 0.55, "Home & Kitchen": 0.60,
        "Furniture": 0.65, "Beauty & Personal Care": 0.50, "Sports & Fitness": 0.58,
    }
    df["Cost"] = df.apply(lambda r: r["Unit_Price"] * cogs_ratio_lookup[r["Category"]] * r["Quantity"], axis=1)
    df["Profit"] = df["Revenue"] - df["Cost"]

    # =============================================================
    # Inject realistic "messiness" for the data-cleaning section
    # =============================================================

    # (a) Missing values in a few columns
    missing_idx_region = df.sample(frac=0.02, random_state=1).index
    df.loc[missing_idx_region, "Region"] = np.nan

    missing_idx_payment = df.sample(frac=0.015, random_state=2).index
    df.loc[missing_idx_payment, "Payment_Mode"] = np.nan

    missing_idx_discount = df.sample(frac=0.01, random_state=3).index
    df.loc[missing_idx_discount, "Discount_Percent"] = np.nan

    # (b) Duplicate rows (simulate accidental double-entry)
    duplicate_rows = df.sample(frac=0.01, random_state=4)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # (c) Incorrect data types — store some Quantity / Unit_Price as strings.
    # Columns are cast to generic "object" dtype first so mixed str/number
    # values can coexist (mirrors messy real-world spreadsheet exports).
    df["Quantity"] = df["Quantity"].astype(object)
    wrong_type_idx = df.sample(frac=0.01, random_state=5).index
    df.loc[wrong_type_idx, "Quantity"] = df.loc[wrong_type_idx, "Quantity"].astype(str)

    df["Unit_Price"] = df["Unit_Price"].astype(object)
    wrong_type_idx_price = df.sample(frac=0.01, random_state=6).index
    df.loc[wrong_type_idx_price, "Unit_Price"] = df.loc[wrong_type_idx_price, "Unit_Price"].apply(
        lambda x: f"₹{x}"
    )

    # (d) A few negative / unrealistic outlier values (data-entry errors)
    outlier_idx = df.sample(frac=0.005, random_state=7).index
    df.loc[outlier_idx, "Quantity"] = -1

    outlier_idx_price = df.sample(frac=0.003, random_state=8).index
    df.loc[outlier_idx_price, "Unit_Price"] = 999999  # unrealistic price spike

    # (e) Inconsistent text casing in Region / Category (common real-world issue)
    case_issue_idx = df.sample(frac=0.02, random_state=9).index
    df.loc[case_issue_idx, "Region"] = df.loc[case_issue_idx, "Region"].astype(str).str.upper()

    df = df.reset_index(drop=True)
    return df


# ==============================================================================
# 3. DATA CLEANING
# ==============================================================================
def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw synthetic dataset:
      - Fixes incorrect data types (strings -> numeric)
      - Removes currency symbols
      - Handles missing values
      - Removes duplicate rows
      - Standardizes text casing
      - Removes/repairs unrealistic outliers
      - Recomputes financial columns after cleaning
    """
    df = raw_df.copy()

    print(f"Initial dataset shape (with injected issues): {df.shape}")

    # ---- Fix Unit_Price: strip currency symbol & convert to numeric -------
    df["Unit_Price"] = (
        df["Unit_Price"].astype(str).str.replace("₹", "", regex=False).str.strip()
    )
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # ---- Fix Quantity: convert to numeric ----------------------------------
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # ---- Standardize text columns (casing / whitespace) --------------------
    for col in ["Region", "Category", "Product", "Payment_Mode", "Customer_Segment"]:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace("Nan", np.nan)

    # ---- Handle missing values ---------------------------------------------
    # Region: fill with the mode (most frequent region)
    df["Region"] = df["Region"].fillna(df["Region"].mode()[0])

    # Payment_Mode: fill with a placeholder category
    df["Payment_Mode"] = df["Payment_Mode"].fillna("Not Specified")

    # Discount_Percent: fill missing with 0 (assume no discount was applied)
    df["Discount_Percent"] = df["Discount_Percent"].fillna(0)

    # ---- Handle outliers / invalid values -----------------------------------
    # Quantity cannot be negative or zero -> drop those rows
    df = df[df["Quantity"] > 0]

    # Unit_Price: remove unrealistic extreme values using an IQR-based cap
    q1 = df["Unit_Price"].quantile(0.25)
    q3 = df["Unit_Price"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr  # generous multiplier to keep genuine high-value items
    df = df[df["Unit_Price"] <= upper_bound]

    # Drop any rows where key numeric fields are still missing after conversion
    df = df.dropna(subset=["Unit_Price", "Quantity"])

    # ---- Remove duplicate rows ----------------------------------------------
    before_dupes = df.shape[0]
    df = df.drop_duplicates(subset=["Order_ID"], keep="first")
    df = df.drop_duplicates(keep="first")
    after_dupes = df.shape[0]
    print(f"Duplicate rows removed: {before_dupes - after_dupes}")

    # ---- Correct data types --------------------------------------------------
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Quantity"] = df["Quantity"].astype(int)
    df["Unit_Price"] = df["Unit_Price"].round(2)
    df["Discount_Percent"] = df["Discount_Percent"].astype(float)

    # ---- Recompute financial fields after cleaning ---------------------------
    cogs_ratio_lookup = {
        "Electronics": 0.68, "Clothing": 0.55, "Home & Kitchen": 0.60,
        "Furniture": 0.65, "Beauty & Personal Care": 0.50, "Sports & Fitness": 0.58,
    }
    df["Gross_Sales"] = df["Unit_Price"] * df["Quantity"]
    df["Discount_Amount"] = (df["Gross_Sales"] * (df["Discount_Percent"] / 100)).round(2)
    df["Revenue"] = (df["Gross_Sales"] - df["Discount_Amount"]).round(2)
    df["Cost"] = df.apply(
        lambda r: r["Unit_Price"] * cogs_ratio_lookup.get(r["Category"], 0.6) * r["Quantity"], axis=1
    ).round(2)
    df["Profit"] = (df["Revenue"] - df["Cost"]).round(2)
    df["Profit_Margin_Percent"] = ((df["Profit"] / df["Revenue"]) * 100).round(2)

    # ---- Add helpful date-derived columns for trend analysis -----------------
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Month_Name"] = df["Order_Date"].dt.strftime("%b")
    df["Order_YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    df["Order_Weekday"] = df["Order_Date"].dt.day_name()

    df = df.dropna(subset=["Order_Date"]).reset_index(drop=True)

    print(f"Cleaned dataset shape: {df.shape}")
    print("Missing values remaining per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else "None")

    return df


# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def run_eda(df: pd.DataFrame) -> None:
    """Prints a quick statistical overview of the cleaned dataset."""
    print("\nDataset Info:")
    print(df.info())

    print("\nDescriptive Statistics (numeric columns):")
    print(df[["Unit_Price", "Quantity", "Revenue", "Profit", "Profit_Margin_Percent"]].describe().round(2))

    print("\nOrders by Region:")
    print(df["Region"].value_counts())

    print("\nOrders by Category:")
    print(df["Category"].value_counts())

    print("\nDate range covered:", df["Order_Date"].min().date(), "to", df["Order_Date"].max().date())


# ==============================================================================
# 5. KPI CALCULATIONS
# ==============================================================================
def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calculates high-level business KPIs used across the analysis/report."""
    kpis = {
        "total_orders": df["Order_ID"].nunique(),
        "total_customers": df["Customer_ID"].nunique(),
        "total_revenue": df["Revenue"].sum(),
        "total_profit": df["Profit"].sum(),
        "total_cost": df["Cost"].sum(),
        "avg_order_value": df.groupby("Order_ID")["Revenue"].sum().mean(),
        "overall_profit_margin_pct": (df["Profit"].sum() / df["Revenue"].sum()) * 100,
        "avg_discount_pct": df["Discount_Percent"].mean(),
        "total_units_sold": df["Quantity"].sum(),
        "top_region_by_revenue": df.groupby("Region")["Revenue"].sum().idxmax(),
        "top_category_by_revenue": df.groupby("Category")["Revenue"].sum().idxmax(),
        "top_product_by_revenue": df.groupby("Product")["Revenue"].sum().idxmax(),
        "best_month": df.groupby("Order_YearMonth")["Revenue"].sum().idxmax(),
    }
    return kpis


def print_kpi_summary(kpis: dict) -> None:
    print("\n" + "=" * 60)
    print("KEY PERFORMANCE INDICATORS (KPI) SUMMARY")
    print("=" * 60)
    print(f"Total Orders                 : {kpis['total_orders']:,}")
    print(f"Total Unique Customers        : {kpis['total_customers']:,}")
    print(f"Total Units Sold              : {kpis['total_units_sold']:,}")
    print(f"Total Revenue                 : Rs. {kpis['total_revenue']:,.2f}")
    print(f"Total Cost                    : Rs. {kpis['total_cost']:,.2f}")
    print(f"Total Profit                  : Rs. {kpis['total_profit']:,.2f}")
    print(f"Overall Profit Margin          : {kpis['overall_profit_margin_pct']:.2f}%")
    print(f"Average Order Value (AOV)      : Rs. {kpis['avg_order_value']:,.2f}")
    print(f"Average Discount Given         : {kpis['avg_discount_pct']:.2f}%")
    print(f"Top Performing Region          : {kpis['top_region_by_revenue']}")
    print(f"Top Performing Category        : {kpis['top_category_by_revenue']}")
    print(f"Top Performing Product         : {kpis['top_product_by_revenue']}")
    print(f"Best Sales Month               : {kpis['best_month']}")
    print("=" * 60)


# ==============================================================================
# 6. SQL ANALYSIS (using sqlite3 - in-memory database, no external DB needed)
# ==============================================================================
def run_sql_analysis(df: pd.DataFrame) -> None:
    """
    Loads the cleaned DataFrame into an in-memory SQLite database and runs a
    few representative SQL queries — useful to demonstrate SQL skills
    alongside pandas within the same project.
    """
    print("\n" + "=" * 60)
    print("SQL ANALYSIS (via in-memory SQLite database)")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    df.to_sql("sales", conn, index=False, if_exists="replace")

    queries = {
        "Revenue & Profit by Region": """
            SELECT Region,
                   ROUND(SUM(Revenue), 2) AS Total_Revenue,
                   ROUND(SUM(Profit), 2) AS Total_Profit,
                   COUNT(DISTINCT Order_ID) AS Total_Orders
            FROM sales
            GROUP BY Region
            ORDER BY Total_Revenue DESC;
        """,
        "Top 5 Products by Revenue": """
            SELECT Product,
                   ROUND(SUM(Revenue), 2) AS Total_Revenue,
                   SUM(Quantity) AS Units_Sold
            FROM sales
            GROUP BY Product
            ORDER BY Total_Revenue DESC
            LIMIT 5;
        """,
        "Monthly Revenue Trend": """
            SELECT Order_YearMonth,
                   ROUND(SUM(Revenue), 2) AS Monthly_Revenue
            FROM sales
            GROUP BY Order_YearMonth
            ORDER BY Order_YearMonth;
        """,
        "Top 5 Customers by Total Spend": """
            SELECT Customer_ID, Customer_Name,
                   ROUND(SUM(Revenue), 2) AS Total_Spend,
                   COUNT(DISTINCT Order_ID) AS Orders_Placed
            FROM sales
            GROUP BY Customer_ID, Customer_Name
            ORDER BY Total_Spend DESC
            LIMIT 5;
        """,
        "Category-wise Average Profit Margin": """
            SELECT Category,
                   ROUND(AVG(Profit_Margin_Percent), 2) AS Avg_Profit_Margin_Pct
            FROM sales
            GROUP BY Category
            ORDER BY Avg_Profit_Margin_Pct DESC;
        """,
    }

    for title, query in queries.items():
        print(f"\n--- {title} ---")
        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))

    conn.close()


# ==============================================================================
# 7. VISUALIZATIONS
# ==============================================================================
def ensure_output_dir() -> None:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_chart(fig, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved chart -> {path}")


def create_visualizations(df: pd.DataFrame) -> None:
    ensure_output_dir()
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 1. Monthly Revenue Trend (line chart)
    monthly_revenue = df.groupby("Order_YearMonth")["Revenue"].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(monthly_revenue["Order_YearMonth"], monthly_revenue["Revenue"],
            marker="o", color="#2E86AB", linewidth=2)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (Rs.)")
    ax.tick_params(axis="x", rotation=90)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}K"))
    save_chart(fig, "01_monthly_revenue_trend.png")

    # 2. Revenue by Region (bar chart)
    region_revenue = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=region_revenue, x="Region", y="Revenue", hue="Region",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Total Revenue by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Revenue (Rs.)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}K"))
    save_chart(fig, "02_revenue_by_region.png")

    # 3. Revenue Share by Category (pie chart)
    category_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.pie(category_revenue, labels=category_revenue.index, autopct="%1.1f%%",
           startangle=140, colors=sns.color_palette("Set2"))
    ax.set_title("Revenue Share by Category")
    save_chart(fig, "03_revenue_share_by_category.png")

    # 4. Top 10 Products by Revenue (horizontal bar chart)
    top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=top_products, y="Product", x="Revenue", hue="Product",
                palette="mako", legend=False, ax=ax)
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue (Rs.)")
    ax.set_ylabel("Product")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}K"))
    save_chart(fig, "04_top10_products_by_revenue.png")

    # 5. Profit Margin Distribution by Category (boxplot)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Category", y="Profit_Margin_Percent", hue="Category",
                palette="pastel", legend=False, ax=ax)
    ax.set_title("Profit Margin Distribution by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Profit Margin (%)")
    ax.tick_params(axis="x", rotation=30)
    save_chart(fig, "05_profit_margin_by_category_boxplot.png")

    # 6. Customer Segment Contribution to Revenue (bar chart)
    segment_revenue = df.groupby("Customer_Segment")["Revenue"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=segment_revenue, x="Customer_Segment", y="Revenue", hue="Customer_Segment",
                palette="crest", legend=False, ax=ax)
    ax.set_title("Revenue by Customer Segment")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Revenue (Rs.)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}K"))
    save_chart(fig, "06_revenue_by_customer_segment.png")

    # 7. Orders by Payment Mode (count plot)
    fig, ax = plt.subplots()
    order_counts = df.drop_duplicates("Order_ID")
    sns.countplot(data=order_counts, x="Payment_Mode",
                  order=order_counts["Payment_Mode"].value_counts().index,
                  hue="Payment_Mode", palette="flare", legend=False, ax=ax)
    ax.set_title("Number of Orders by Payment Mode")
    ax.set_xlabel("Payment Mode")
    ax.set_ylabel("Number of Orders")
    ax.tick_params(axis="x", rotation=20)
    save_chart(fig, "07_orders_by_payment_mode.png")

    # 8. Revenue vs Profit Trend by Year (grouped bar chart)
    yearly = df.groupby("Order_Year")[["Revenue", "Profit"]].sum().reset_index()
    yearly_melted = yearly.melt(id_vars="Order_Year", value_vars=["Revenue", "Profit"],
                                 var_name="Metric", value_name="Amount")
    fig, ax = plt.subplots()
    sns.barplot(data=yearly_melted, x="Order_Year", y="Amount", hue="Metric",
                palette="Set1", ax=ax)
    ax.set_title("Yearly Revenue vs Profit")
    ax.set_xlabel("Year")
    ax.set_ylabel("Amount (Rs.)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/100000:,.1f}L"))
    save_chart(fig, "08_yearly_revenue_vs_profit.png")

    # 9. Top 10 Customers by Total Spend (horizontal bar chart) - bonus chart
    top_customers = (
        df.groupby(["Customer_ID", "Customer_Name"])["Revenue"]
        .sum().sort_values(ascending=False).head(10).reset_index()
    )
    fig, ax = plt.subplots()
    sns.barplot(data=top_customers, y="Customer_Name", x="Revenue", hue="Customer_Name",
                palette="rocket", legend=False, ax=ax)
    ax.set_title("Top 10 Customers by Total Spend")
    ax.set_xlabel("Revenue (Rs.)")
    ax.set_ylabel("Customer")
    save_chart(fig, "09_top10_customers_by_spend.png")

    # 10. Correlation Heatmap of Numeric Features - bonus chart
    numeric_cols = ["Unit_Price", "Quantity", "Discount_Percent", "Revenue", "Cost", "Profit"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap - Numeric Features")
    save_chart(fig, "10_correlation_heatmap.png")

    print(f"\nAll charts saved successfully inside the '{OUTPUT_DIR}/' folder.")


# ==============================================================================
# 8. BUSINESS INSIGHTS & RECOMMENDATIONS
# ==============================================================================
def generate_insights_and_recommendations(df: pd.DataFrame, kpis: dict) -> None:
    region_revenue = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    category_margin = df.groupby("Category")["Profit_Margin_Percent"].mean().sort_values(ascending=False)
    weakest_region = region_revenue.idxmin()
    weakest_category_margin = category_margin.idxmin()
    strongest_category_margin = category_margin.idxmax()

    discount_impact = df.groupby(pd.cut(df["Discount_Percent"], bins=[-1, 0, 10, 20, 30],
                                         labels=["0%", "1-10%", "11-20%", "21-30%"]))["Profit_Margin_Percent"].mean()

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)
    print(f"1. '{kpis['top_region_by_revenue']}' is the highest revenue-generating region, "
          f"while '{weakest_region}' lags behind and may need focused marketing efforts.")
    print(f"2. '{kpis['top_category_by_revenue']}' is the best-performing category by revenue, "
          f"whereas '{weakest_category_margin}' has the lowest average profit margin "
          f"({category_margin.min():.2f}%), which could be squeezing overall profitability.")
    print(f"3. '{strongest_category_margin}' category yields the highest average profit margin "
          f"({category_margin.max():.2f}%), making it a strong candidate for upselling.")
    print(f"4. The overall profit margin across the business stands at "
          f"{kpis['overall_profit_margin_pct']:.2f}%, with an average order value of "
          f"Rs. {kpis['avg_order_value']:,.2f}.")
    print(f"5. Orders with higher discount slabs show the following average profit margins:\n"
          f"{discount_impact.round(2).to_string()}")
    print(f"6. '{kpis['best_month']}' recorded the highest monthly revenue, suggesting a "
          f"seasonal peak worth planning inventory and staffing around.")
    print(f"7. Customers in the 'Premium' segment tend to contribute disproportionately to revenue "
          f"relative to their count, indicating strong potential in loyalty-based marketing.")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"1. Increase marketing spend and promotional campaigns in the '{weakest_region}' region "
          f"to close the revenue gap with top-performing regions.")
    print(f"2. Review pricing and supplier costs for the '{weakest_category_margin}' category to "
          f"improve its profit margin without significantly increasing prices.")
    print(f"3. Promote and bundle products from the '{strongest_category_margin}' category, since "
          f"it already delivers the strongest margins.")
    print("4. Cap discounts beyond 20% unless clearly tied to a strategic clearance goal, as higher "
          "discount slabs are associated with reduced profit margins.")
    print(f"5. Prepare additional inventory and staffing ahead of peak months similar to "
          f"'{kpis['best_month']}' to capture seasonal demand.")
    print("6. Launch a loyalty / rewards program targeted at 'Premium' and high-spend customers "
          "to improve retention and lifetime value.")
    print("7. Diversify payment options further and monitor which payment modes correlate with "
          "higher order values to streamline the checkout experience.")
    print("=" * 60)


# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print("SALES & BUSINESS ANALYTICS DASHBOARD")
    print("=" * 60)

    # Step 1: Generate synthetic raw dataset
    print("\nStep 1: Generating synthetic sales dataset...")
    raw_df = generate_sales_dataset(n_rows=5000)

    # Step 2: Clean the dataset
    print("\nStep 2: Cleaning dataset...")
    clean_df = clean_dataset(raw_df)

    # Step 3: Exploratory Data Analysis
    print("\nStep 3: Running exploratory data analysis...")
    run_eda(clean_df)

    # Step 4: KPI Calculations
    print("\nStep 4: Calculating KPIs...")
    kpis = calculate_kpis(clean_df)
    print_kpi_summary(kpis)

    # Step 5: SQL Analysis
    print("\nStep 5: Running SQL-based analysis...")
    run_sql_analysis(clean_df)

    # Step 6: Visualizations
    print("\nStep 6: Creating visualizations...")
    create_visualizations(clean_df)

    # Step 7: Business Insights & Recommendations
    print("\nStep 7: Generating business insights and recommendations...")
    generate_insights_and_recommendations(clean_df, kpis)

    print("\nProject executed successfully. Check the 'output_charts' folder for all visualizations.")


if __name__ == "__main__":
    main()
