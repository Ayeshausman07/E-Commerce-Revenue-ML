"""
E-Commerce Sales Analytics & Revenue Prediction Dashboard
Created by: Ayesha Usman

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RevIQ · Ayesha Usman",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ─────────────────────── */
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

html, body, [class*="css"] {
    background-color: #f8f8f8 !important;
    color: #111827;
}
.main, [data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #f8f8f8 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1300px; }

/* ── Banner ───────────────────────────── */
.banner {
    background: #ffffff;
    border-radius: 20px;
    padding: 48px 56px;
    margin-bottom: 36px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
}
.banner-accent {
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: #dc2626;
    border-radius: 20px 0 0 20px;
}
.banner-tag {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #dc2626;
    margin-bottom: 14px;
}
.banner-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
    margin: 0 0 12px;
    letter-spacing: -0.5px;
}
.banner-title span { color: #dc2626; }
.banner-desc {
    font-size: 15px;
    font-weight: 400;
    color: #6b7280;
    max-width: 520px;
    line-height: 1.6;
}
.banner-row {
    display: flex;
    gap: 12px;
    margin-top: 24px;
    align-items: center;
}
.badge {
    font-size: 12px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 100px;
}
.badge-dark {
    background: #111827;
    color: #ffffff;
}
.badge-red {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}

/* ── KPI Grid ─────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 16px;
    padding: 24px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    box-shadow: 0 6px 28px rgba(220,38,38,0.1);
    border-color: #fca5a5;
}
.kpi-card-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: #dc2626;
    border-radius: 0 0 16px 16px;
    opacity: 0;
    transition: opacity 0.2s;
}
.kpi-card:hover .kpi-card-bar { opacity: 1; }
.kpi-icon {
    font-size: 22px;
    margin-bottom: 12px;
    line-height: 1;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #111827;
    line-height: 1;
    letter-spacing: -0.5px;
}
.kpi-sub {
    font-size: 12px;
    color: #9ca3af;
    font-weight: 400;
    margin-top: 6px;
}

/* ── Section Header ───────────────────── */
.sec-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #dc2626;
    margin-bottom: 6px;
    margin-top: 36px;
}
.sec-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #111827;
    margin: 0 0 20px;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #f0f0f0;
}

/* ── Info Box ─────────────────────────── */
.info-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 3px solid #dc2626;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 13.5px;
    font-weight: 400;
    color: #374151;
    line-height: 1.7;
}
.info-box b, .info-box strong { color: #dc2626; font-weight: 600; }

/* ── Metric Card ──────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #f0f0f0;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    border-color: #fca5a5;
    box-shadow: 0 4px 16px rgba(220,38,38,0.08);
}
.metric-card .mc-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1;
}
.metric-card .mc-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 8px;
}

/* ── Prediction Result ────────────────── */
.pred-card {
    background: #dc2626;
    border-radius: 20px;
    padding: 44px;
    text-align: center;
    margin: 24px 0;
    box-shadow: 0 8px 40px rgba(220,38,38,0.25);
}
.pred-card .p-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.7);
    margin-bottom: 10px;
}
.pred-card .p-amount {
    font-size: 4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -2px;
    line-height: 1;
}
.pred-card .p-range {
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin-top: 10px;
    font-weight: 400;
}

/* ── Sidebar ──────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #f0f0f0 !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 500 !important;
     color: #111827 !important; 
    padding: 9px 0 !important;
}
.sb-logo {
    font-size: 1.9rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
}
.sb-logo span { color: #dc2626; }
.sb-tagline {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 2px;
    margin-bottom: 24px;
}
.sb-divider {
    height: 1px;
    background: #f0f0f0;
    margin: 12px 0 16px;
}
.sb-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #dc2626;
    margin: 16px 0 10px;
}
            /* Force dark color on radio labels in sidebar */
[data-testid="stSidebar"] .st-emotion-cache-1v0mbdj {
    color: #111827 !important;
}

div[role="radiogroup"] label div {
    color: #1f2937 !important;
    font-weight: 500 !important;
}
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f9f9f9;
    font-size: 13px;
    color: #6b7280;
    font-weight: 400;
}
.sb-stat b {
    color: #111827;
    font-weight: 700;
    font-size: 12px;
}
.sb-cat {
    font-size: 13px;
    color: #374151;
    padding: 7px 0;
    border-bottom: 1px solid #f9f9f9;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 400;
}
.sb-cat::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #dc2626;
    flex-shrink: 0;
}

/* ── Tabs ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #f0f0f0;
    gap: 2px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b7280;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 22px;
}
.stTabs [aria-selected="true"] {
    background: #dc2626 !important;
    color: #ffffff !important;
}

/* ── Buttons ──────────────────────────── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 11px 28px !important;
    box-shadow: 0 2px 12px rgba(220,38,38,0.25) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: #b91c1c !important;
    box-shadow: 0 6px 20px rgba(220,38,38,0.32) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ───────────────────────────── */
.stNumberInput input {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* ── Dataframe ─────────────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #f0f0f0 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}

/* ── Success ──────────────────────────── */
.stSuccess {
    background: #fef2f2 !important;
    border-color: #fca5a5 !important;
    color: #dc2626 !important;
    border-radius: 10px !important;
}

/* ── Column desc ──────────────────────── */
.col-row {
    display: flex;
    gap: 18px;
    padding: 12px 8px;
    border-bottom: 1px solid #f9f9f9;
    font-size: 13px;
    align-items: flex-start;
    border-radius: 6px;
    transition: background 0.15s;
}
.col-row:hover { background: #fef2f2; }
.col-row .c-name {
    font-size: 12px;
    font-weight: 700;
    color: #dc2626;
    min-width: 165px;
}
.col-row .c-type {
    font-size: 11px;
    color: #d1d5db;
    min-width: 65px;
    font-weight: 500;
    margin-top: 1px;
}
.col-row .c-desc {
    color: #6b7280;
    line-height: 1.5;
    font-weight: 400;
}

/* ── Scrollbar ────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f8f8f8; }
::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #dc2626; }

hr {
    border: none !important;
    border-top: 1px solid #f0f0f0 !important;
    margin: 24px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHART THEME
# ─────────────────────────────────────────────
RED     = "#dc2626"
PALETTE = ["#dc2626", "#374151", "#6b7280", "#9ca3af", "#fca5a5", "#1f2937"]
BG      = "#ffffff"
TEXT    = "#111827"
GRID    = "#f3f4f6"
SUBTEXT = "#6b7280"

def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=SUBTEXT, labelsize=10)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color("#e5e7eb")
    ax.grid(color=GRID, linestyle='-', linewidth=0.8, alpha=1)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12.5, fontweight='bold', pad=14)


# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('ecommerce_sales_analytics_5000.csv')

@st.cache_resource
def train_model(data):
    features = ['quantity', 'unit_price', 'discount', 'delivery_days', 'customer_rating']
    X, y = data[features], data['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    poly   = PolynomialFeatures(degree=2)
    Xtr = poly.fit_transform(scaler.fit_transform(X_train))
    Xte = poly.transform(scaler.transform(X_test))
    model = LinearRegression().fit(Xtr, y_train)
    yp_tr, yp_te = model.predict(Xtr), model.predict(Xte)
    return model, {
        'train_r2': r2_score(y_train, yp_tr),
        'test_r2':  r2_score(y_test,  yp_te),
        'mae':      mean_absolute_error(y_test, yp_te),
        'rmse':     np.sqrt(mean_squared_error(y_test, yp_te)),
        'scaler': scaler, 'poly': poly,
        'y_test': y_test, 'y_pred': yp_te
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    data = load_data()

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div class="sb-logo">Rev<span>IQ</span></div>
            <div class="sb-tagline">E-Commerce Intelligence</div>
            <div class="sb-divider"></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("", ["Data Overview","Visual Analytics","ML Model","Revenue Predictor"],
                        label_visibility="collapsed")

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Dataset Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="sb-stat">Records <b>{len(data):,}</b></div>
            <div class="sb-stat">Features <b>{len(data.columns)}</b></div>
            <div class="sb-stat">Total Revenue <b>${data['revenue'].sum()/1e6:.1f}M</b></div>
            <div class="sb-stat">Avg Order <b>${data['revenue'].mean():,.0f}</b></div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">Categories</div>', unsafe_allow_html=True)
        for cat in data['product_category'].unique():
            st.markdown(f'<div class="sb-cat">{cat}</div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="sb-divider"></div>
            <div style="font-size:11px;color:#d1d5db;font-weight:500;padding-top:2px;">
                Made by Ayesha Usman
            </div>
        """, unsafe_allow_html=True)

    # ── Banner ────────────────────────────────
    st.markdown("""
        <div class="banner">
            <div class="banner-accent"></div>
            <div class="banner-tag">Machine Learning Project</div>
            <div class="banner-title">E-Commerce <span>Revenue</span><br>Intelligence Dashboard</div>
            <div class="banner-desc">Sales analytics and revenue prediction using Polynomial Regression — built for data-driven decision making.</div>
            <div class="banner-row">
                <div class="badge badge-dark">◇By Ayesha Usman</div>
                <div class="badge badge-red">ML Dashboard</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if page == "Data Overview":       show_data_overview(data)
    elif page == "Visual Analytics":  show_visual_analytics(data)
    elif page == "ML Model":          show_ml_model(data)
    elif page == "Revenue Predictor": show_revenue_predictor(data)


# ─────────────────────────────────────────────
#  PAGE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────
def show_data_overview(data):
    st.markdown('<div class="sec-label">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📋 Data Overview</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">${data['revenue'].sum()/1e6:.1f}M</div>
            <div class="kpi-sub">Across all orders</div>
            <div class="kpi-card-bar"></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🛒</div>
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">${data['revenue'].mean():,.0f}</div>
            <div class="kpi-sub">Per transaction</div>
            <div class="kpi-card-bar"></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⭐</div>
            <div class="kpi-label">Avg Rating</div>
            <div class="kpi-value">{data['customer_rating'].mean():.2f}</div>
            <div class="kpi-sub">Out of 5.0</div>
            <div class="kpi-card-bar"></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏷️</div>
            <div class="kpi-label">Categories</div>
            <div class="kpi-value">{data['product_category'].nunique()}</div>
            <div class="kpi-sub">Distinct product groups</div>
            <div class="kpi-card-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
        if st.button("📥 Show Full Dataset"):
            st.dataframe(data, use_container_width=True)
    with col2:
        st.markdown('<div class="sec-label">Statistics</div>', unsafe_allow_html=True)
        st.dataframe(data.describe().round(2), use_container_width=True)
        if data.isnull().sum().sum() == 0:
            st.success("✓ No missing values — dataset is clean.")
        else:
            st.warning(f"{data.isnull().sum().sum()} missing values found.")

    st.markdown('<div class="sec-label">Schema</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📌 Column Definitions</div>', unsafe_allow_html=True)

    cols_info = [
        ("order_id",         "object",  "Unique identifier for each order"),
        ("order_date",       "object",  "Date the customer placed the order"),
        ("customer_id",      "object",  "Unique ID assigned to each customer"),
        ("product_category", "object",  "Product type — Electronics, Clothing, Beauty, or Home"),
        ("region",           "object",  "Customer location — North, South, East, or West"),
        ("quantity",         "int",     "Number of items ordered"),
        ("unit_price",       "float",   "Price of a single item in USD ($)"),
        ("discount",         "float",   "Discount applied (0 = none, 0.5 = 50% off)"),
        ("payment_method",   "object",  "Payment type — Card, Wallet, or Cash on Delivery"),
        ("delivery_days",    "int",     "Days taken to deliver the order"),
        ("customer_rating",  "float",   "Customer satisfaction score, 1 (poor) to 5 (excellent)"),
        ("revenue",          "float",   "Total revenue from the order — this is our prediction target"),
    ]
    for name, dtype, desc in cols_info:
        st.markdown(f"""
        <div class="col-row">
            <div class="c-name">{name}</div>
            <div class="c-type">{dtype}</div>
            <div class="c-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE 2 — VISUAL ANALYTICS
# ─────────────────────────────────────────────
def show_visual_analytics(data):
    st.markdown('<div class="sec-label">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📊 Visual Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Revenue Distribution", "Category & Region",
        "Correlation Analysis", "Box Plot Analysis"
    ])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.histplot(data['revenue'], kde=True, bins=50, color=RED,
                         edgecolor="white", alpha=0.65, ax=ax)
            ax.lines[0].set_color("#374151")
            ax.lines[0].set_linewidth(2.2)
            style_ax(ax, "Revenue Distribution")
            ax.set_xlabel("Revenue ($)")
            ax.set_ylabel("Number of Orders")
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown('<div class="info-box"><strong>What this shows:</strong> The spread of revenue across all orders. The curve shows where most revenue values cluster.</div>', unsafe_allow_html=True)
        with col2:
            for lbl, v in [("Mean", data['revenue'].mean()), ("Median", data['revenue'].median()),
                           ("Std Dev", data['revenue'].std()), ("Min", data['revenue'].min()),
                           ("Max", data['revenue'].max())]:
                st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><div class="mc-val">${v:,.0f}</div><div class="mc-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cat_rev = data.groupby('product_category')['revenue'].mean().sort_values()
            colors  = [RED if i == len(cat_rev)-1 else "#e5e7eb" for i in range(len(cat_rev))]
            ax.barh(cat_rev.index, cat_rev.values, color=colors, edgecolor="white", height=0.5)
            style_ax(ax, "Avg Revenue by Category")
            ax.set_xlabel("Average Revenue ($)")
            fig.tight_layout(); st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cat_counts = data['product_category'].value_counts()
            pie_colors = [RED, "#374151", "#9ca3af", "#d1d5db"]
            wedges, texts, autotexts = ax.pie(
                cat_counts.values, labels=cat_counts.index,
                autopct='%1.1f%%', colors=pie_colors,
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.58, edgecolor="white", linewidth=2))
            for t in texts:     t.set_color(TEXT); t.set_fontsize(11)
            for t in autotexts: t.set_color("white"); t.set_fontsize(10); t.set_fontweight('bold')
            ax.set_facecolor(BG); ax.figure.patch.set_facecolor(BG)
            ax.set_title("Order Count by Category", color=TEXT, fontsize=12.5, fontweight='bold', pad=14)
            fig.tight_layout(); st.pyplot(fig)

        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            reg_rev = data.groupby('region')['revenue'].sum().sort_values()
            colors  = [RED if i == len(reg_rev)-1 else "#e5e7eb" for i in range(len(reg_rev))]
            ax.bar(reg_rev.index, reg_rev.values, color=colors, edgecolor="white", width=0.48)
            style_ax(ax, "Total Revenue by Region")
            ax.set_ylabel("Total Revenue ($)")
            fig.tight_layout(); st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            pay_rev  = data.groupby('payment_method')['revenue'].sum()
            p_colors = [RED, "#374151", "#9ca3af"]
            wedges, texts, autotexts = ax.pie(
                pay_rev.values, labels=pay_rev.index,
                autopct='%1.1f%%', colors=p_colors,
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.58, edgecolor="white", linewidth=2))
            for t in texts:     t.set_color(TEXT); t.set_fontsize(11)
            for t in autotexts: t.set_color("white"); t.set_fontsize(10); t.set_fontweight('bold')
            ax.set_facecolor(BG); ax.figure.patch.set_facecolor(BG)
            ax.set_title("Revenue by Payment Method", color=TEXT, fontsize=12.5, fontweight='bold', pad=14)
            fig.tight_layout(); st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns([3, 2])
        with col1:
            num_cols = data.select_dtypes(include=[np.number]).columns
            corr = data[num_cols].corr()
            fig, ax = plt.subplots(figsize=(9, 7))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.light_palette(RED, as_cmap=True)
            sns.heatmap(corr, annot=True, cmap=cmap, center=0, fmt='.2f',
                        mask=mask, square=True, linewidths=1.5,
                        linecolor="#f8f8f8", ax=ax,
                        annot_kws={'size': 9, 'color': TEXT},
                        cbar_kws={'shrink': 0.8})
            ax.set_facecolor(BG); ax.figure.patch.set_facecolor(BG)
            ax.tick_params(colors=SUBTEXT, labelsize=10)
            ax.set_title("Correlation Matrix", color=TEXT, fontsize=12.5, fontweight='bold', pad=16)
            fig.tight_layout(); st.pyplot(fig)
            st.markdown('<div class="info-box"><strong>How to read:</strong> Values near <strong>+1</strong> = features move together. Near <strong>−1</strong> = they move opposite. Near <strong>0</strong> = no relationship.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="sec-label">Revenue Correlations</div>', unsafe_allow_html=True)
            rev_corr = corr['revenue'].drop('revenue').sort_values(ascending=False)
            for cname, val in rev_corr.items():
                pct   = int(abs(val) * 100)
                color = RED if val > 0 else "#374151"
                st.markdown(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:5px;">
                        <span style="color:#374151;">{cname}</span>
                        <span style="color:{color};">{val:+.3f}</span>
                    </div>
                    <div style="background:#f3f4f6;border-radius:4px;height:7px;overflow:hidden;">
                        <div style="width:{pct}%;background:{color};height:7px;border-radius:4px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:24px;">Scatter Plots</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.scatter(data['quantity'], data['revenue'], alpha=0.3, c=RED, s=14, edgecolors='none')
            style_ax(ax, "Revenue vs Quantity")
            ax.set_xlabel("Quantity"); ax.set_ylabel("Revenue ($)")
            fig.tight_layout(); st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.scatter(data['unit_price'], data['revenue'], alpha=0.3, c="#374151", s=14, edgecolors='none')
            style_ax(ax, "Revenue vs Unit Price")
            ax.set_xlabel("Unit Price ($)"); ax.set_ylabel("Revenue ($)")
            fig.tight_layout(); st.pyplot(fig)

    with tab4:
        st.markdown('<div class="info-box"><strong>Box plots</strong> show the spread of revenue per group. The box holds most values, the middle line is the median, and dots outside are outliers.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cats = data['product_category'].unique()
            cp   = {c: (RED if i == 0 else PALETTE[i]) for i, c in enumerate(cats)}
            sns.boxplot(x='product_category', y='revenue', data=data,
                        palette=cp, ax=ax, linecolor="#374151", linewidth=0.8,
                        flierprops=dict(marker='o', markerfacecolor=RED, markersize=3, alpha=0.4))
            style_ax(ax, "Revenue by Category")
            ax.set_xlabel("Category"); ax.set_ylabel("Revenue ($)")
            plt.xticks(rotation=15); fig.tight_layout(); st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            regs = data['region'].unique()
            rp   = {r: (RED if i == 0 else PALETTE[i]) for i, r in enumerate(regs)}
            sns.boxplot(x='region', y='revenue', data=data,
                        palette=rp, ax=ax, linecolor="#374151", linewidth=0.8,
                        flierprops=dict(marker='o', markerfacecolor=RED, markersize=3, alpha=0.4))
            style_ax(ax, "Revenue by Region")
            ax.set_xlabel("Region"); ax.set_ylabel("Revenue ($)")
            fig.tight_layout(); st.pyplot(fig)


# ─────────────────────────────────────────────
#  PAGE 3 — ML MODEL
# ─────────────────────────────────────────────
def show_ml_model(data):
    st.markdown('<div class="sec-label">Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🧠 Model Performance</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">This app uses a <strong>Polynomial Regression</strong> model (degree 2) to predict revenue. It captures non-linear patterns, making it more accurate than basic linear regression for real-world data.</div>', unsafe_allow_html=True)

    model, metrics = train_model(data)

    col1, col2, col3, col4 = st.columns(4)
    for col, lbl, val, fmt in zip(
        [col1, col2, col3, col4],
        ["Train R² Score", "Test R² Score", "Mean Abs. Error", "Root Mean Sq. Error"],
        [metrics['train_r2'], metrics['test_r2'], metrics['mae'], metrics['rmse']],
        ["{:.4f}", "{:.4f}", "${:,.2f}", "${:,.2f}"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="mc-val">{fmt.format(val)}</div><div class="mc-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(metrics['y_test'], metrics['y_pred'],
                   alpha=0.3, c=RED, s=16, edgecolors='none', zorder=3)
        mn = min(metrics['y_test'].min(), metrics['y_pred'].min())
        mx = max(metrics['y_test'].max(), metrics['y_pred'].max())
        ax.plot([mn,mx],[mn,mx], color="#374151", lw=1.8, linestyle='--',
                label='Perfect prediction', zorder=4)
        style_ax(ax, "Actual vs Predicted Revenue")
        ax.set_xlabel("Actual Revenue ($)"); ax.set_ylabel("Predicted Revenue ($)")
        ax.legend(framealpha=0, labelcolor=SUBTEXT, fontsize=10)
        fig.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="info-box">Points close to the dashed line = accurate predictions. The tighter the scatter, the better the model.</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        scores = [metrics['train_r2'], metrics['test_r2']]
        bars   = ax.bar(['Training Score', 'Testing Score'], scores,
                        color=[RED, "#374151"], edgecolor="white", width=0.4)
        ax.set_ylim(0, 1.18)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, s + 0.025,
                    f'{s:.4f}', ha='center', va='bottom',
                    color=TEXT, fontweight='bold', fontsize=12)
        style_ax(ax, "Train vs Test R² Score")
        ax.set_ylabel("R² Score (1.0 = perfect)")
        fig.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="info-box"><strong>R² Score</strong> measures how well the model explains revenue variance. Similar train & test scores mean the model generalises well.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ How It Works</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-box" style="line-height:2.1;">
        <strong>Step 1 — Features:</strong> quantity, unit price, discount, delivery days, rating<br>
        <strong>Step 2 — Normalise:</strong> StandardScaler ensures equal weight across features<br>
        <strong>Step 3 — Polynomial Expansion:</strong> Degree-2 terms capture curved patterns<br>
        <strong>Step 4 — Train:</strong> Model learns from 80% of data<br>
        <strong>Step 5 — Evaluate:</strong> Accuracy tested on the remaining 20%
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="info-box" style="line-height:2.1;">
        <strong>R² (Test):</strong> {metrics['test_r2']:.4f} — explains {metrics['test_r2']*100:.1f}% of variance<br>
        <strong>MAE:</strong> {metrics['mae']:,.2f} — average prediction error per order<br>
        <strong>RMSE:</strong> {metrics['rmse']:,.2f} — penalises large errors more than MAE<br>
        <strong>Algorithm:</strong> Polynomial Regression (degree 2) via scikit-learn<br>
        <strong>Split:</strong> 80% train / 20% test, random_state = 42
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 4 — REVENUE PREDICTOR
# ─────────────────────────────────────────────
def show_revenue_predictor(data):
    st.markdown('<div class="sec-label">Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🎯 Revenue Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Fill in the order details below. The ML model will estimate the expected revenue based on patterns learned from 5,000 past transactions.</div>', unsafe_allow_html=True)

    model, metrics = train_model(data)

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            quantity   = st.number_input("Quantity (units ordered)", min_value=1, max_value=50, value=5)
            unit_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=600.0,
                                         value=150.0, step=10.0)
        with col2:
            discount_percent = st.slider("Discount (%)", min_value=0, max_value=50,
                                        value=15, step=1, format="%d%%")
            delivery_days = st.number_input("Delivery Days", min_value=1, max_value=15, value=5)
        with col3:
            customer_rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0,
                                        value=3.5, step=0.1)
        submitted = st.form_submit_button("⚡ Predict Revenue", type="primary", use_container_width=True)

    if submitted:
        # Convert percentage to decimal for calculations
        discount_decimal = discount_percent / 100
        
        inp  = pd.DataFrame({
            'quantity': [quantity],
            'unit_price': [unit_price],
            'discount': [discount_decimal],
            'delivery_days': [delivery_days],
            'customer_rating': [customer_rating]
        })
        pred  = model.predict(metrics['poly'].transform(metrics['scaler'].transform(inp)))[0]
        lower = pred - metrics['mae']
        upper = pred + metrics['mae']

        st.markdown(f"""
        <div class="pred-card">
            <div class="p-label">Estimated Revenue</div>
            <div class="p-amount">${pred:,.2f}</div>
            <div class="p-range">Likely range: ${lower:,.2f} — ${upper:,.2f}</div>
        </div>""", unsafe_allow_html=True)

        raw   = quantity * unit_price
        disc  = raw * discount_decimal
        after = raw - disc

        col1, col2, col3 = st.columns(3)
        for col, lbl, val in zip([col1,col2,col3],
            ["Raw Total", "Discount Applied", "After Discount"], [raw, disc, after]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="mc-val">${val:,.2f}</div><div class="mc-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ["Raw Total", "After Discount", "ML Prediction"]
        values = [raw, after, pred]
        colors = ["#e5e7eb", "#374151", RED]
        bars   = ax.bar(labels, values, color=colors, edgecolor="white", width=0.44)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.012,
                    f'${val:,.0f}', ha='center', va='bottom',
                    color=TEXT, fontweight='bold', fontsize=11)
        style_ax(ax, "Revenue Breakdown")
        ax.set_ylabel("Amount ($)")
        fig.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="info-box"><strong>Note:</strong> The ML prediction may differ from the simple discount calculation because the model also factors in delivery time and customer rating.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()