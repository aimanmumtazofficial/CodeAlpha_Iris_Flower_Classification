# ================================================
#   Iris Flower Classification - Streamlit App
#   CodeAlpha Data Science Internship - Task 1
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# ------------------------------------------------
# Dark Theme CSS
# ------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0e0e14;
            color: #e8e6f0;
        }

        .stApp {
            background-color: #0e0e14;
        }

        .main-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #e879b0, #9b59f5, #5bbfde);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 0.02em;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #6e6a85;
            font-size: 0.92rem;
            margin-bottom: 2rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .section-header {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #e879b0;
            border-bottom: 1px solid #2a2740;
            padding-bottom: 0.4rem;
            margin-top: 1.8rem;
            margin-bottom: 1rem;
            letter-spacing: 0.03em;
        }

        .metric-card {
            background: linear-gradient(135deg, #1a1730, #221e3a);
            border-radius: 14px;
            padding: 1.4rem 1rem;
            text-align: center;
            border: 1px solid #2e2a4a;
            box-shadow: 0 4px 20px rgba(232, 121, 176, 0.08);
        }

        .metric-value {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #e879b0, #9b59f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-label {
            font-size: 0.78rem;
            color: #6e6a85;
            margin-top: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-box {
            background: linear-gradient(135deg, #1a1730, #1e1535);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            border: 1px solid #3d2f6e;
            text-align: center;
            box-shadow: 0 8px 32px rgba(155, 89, 245, 0.15);
        }

        .prediction-species {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #9b59f5, #5bbfde);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0812, #12101e) !important;
            border-right: 1px solid #2a2740 !important;
        }

        div[data-testid="stSidebar"] > div {
            background: transparent !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0812, #12101e) !important;
        }

        div[data-testid="stSidebar"] * {
            color: #c8c4e0 !important;
        }

        div[data-testid="stSidebar"] .stRadio label {
            color: #c8c4e0 !important;
        }

        div[data-testid="stSidebar"] .stMarkdown {
            color: #c8c4e0 !important;
        }

        .stDataFrame {
            background-color: #1a1730;
            border-radius: 10px;
        }

        div[data-testid="stMetric"] {
            background: #1a1730;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            border: 1px solid #2e2a4a;
        }

        .stProgress > div > div {
            background: linear-gradient(90deg, #e879b0, #9b59f5);
            border-radius: 10px;
        }

        hr {
            border-color: #2a2740;
        }

        .stAlert {
            background-color: #1a1730;
            border: 1px solid #2e2a4a;
            color: #c8c4e0;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Matplotlib Dark Theme
# ------------------------------------------------
plt.rcParams.update({
    'figure.facecolor'  : '#12101e',
    'axes.facecolor'    : '#1a1730',
    'axes.edgecolor'    : '#2e2a4a',
    'axes.labelcolor'   : '#c8c4e0',
    'xtick.color'       : '#c8c4e0',
    'ytick.color'       : '#c8c4e0',
    'text.color'        : '#e8e6f0',
    'grid.color'        : '#2a2740',
    'grid.alpha'        : 0.5,
    'axes.titlecolor'   : '#e8e6f0',
    'figure.edgecolor'  : '#12101e',
})

COLORS = ['#e879b0', '#9b59f5', '#5bbfde']

# ------------------------------------------------
# Load and Prepare Data
# ------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Iris.csv')
    original_count = len(df)
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    clean_frames = []
    for species in df['Species'].unique():
        subset = df[df['Species'] == species].copy()
        for feature in features:
            Q1  = subset[feature].quantile(0.25)
            Q3  = subset[feature].quantile(0.75)
            IQR = Q3 - Q1
            lb  = Q1 - 1.5 * IQR
            ub  = Q3 + 1.5 * IQR
            subset = subset[(subset[feature] >= lb) & (subset[feature] <= ub)]
        clean_frames.append(subset)

    df_clean = pd.concat(clean_frames).reset_index(drop=True)
    outliers_removed = original_count - len(df_clean)
    return df, df_clean, features, outliers_removed


@st.cache_resource
def train_model(df_clean):
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df_clean[features]
    y = df_clean['Species']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm       = confusion_matrix(y_test, y_pred)

    return model, le, accuracy, report, cm, X_test, y_test, y_pred


# ------------------------------------------------
# App Header
# ------------------------------------------------
st.markdown('<div class="main-title">🌸 Iris Flower Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CodeAlpha Data Science Internship &nbsp;·&nbsp; Task 1</div>', unsafe_allow_html=True)
st.markdown("---")

try:
    df_original, df_clean, features, outliers_removed = load_and_prepare_data()
    model, le, accuracy, report, cm, X_test, y_test, y_pred = train_model(df_clean)
except FileNotFoundError:
    st.error("Iris.csv not found. Please place Iris.csv in the same folder as app.py and restart.")
    st.stop()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.markdown("## 🌸 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "Visualizations", "Outlier Handling", "Model Results", "Predict a Flower"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.markdown(f"**Total Rows:** {len(df_original)}")
st.sidebar.markdown(f"**After Cleaning:** {len(df_clean)}")
st.sidebar.markdown(f"**Outliers Removed:** {outliers_removed}")
st.sidebar.markdown(f"**Model Accuracy:** {accuracy * 100:.2f}%")


# ================================================
# Page 1 — Dataset Overview
# ================================================
if page == "Dataset Overview":

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">150</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Species</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{outliers_removed}</div><div class="metric-label">Outliers Removed</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">First 5 Rows</div>', unsafe_allow_html=True)
    st.dataframe(df_original.head(), use_container_width=True)

    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_original.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Species Count</div>', unsafe_allow_html=True)
        species_df = df_original['Species'].value_counts().reset_index()
        species_df.columns = ['Species', 'Count']
        st.dataframe(species_df, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        missing = df_original.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        st.dataframe(missing, use_container_width=True)
        if df_original.isnull().sum().sum() == 0:
            st.success("No missing values found in the dataset.")


# ================================================
# Page 2 — Visualizations
# ================================================
elif page == "Visualizations":

    st.markdown('<div class="section-header">Species Distribution</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    df_original['Species'].value_counts().plot(kind='bar', color=COLORS, edgecolor='#2e2a4a', ax=ax1)
    ax1.set_title('Iris Species Distribution', fontsize=13, fontweight='bold', color='#e8e6f0')
    ax1.set_xlabel('Species', fontsize=10, color='#c8c4e0')
    ax1.set_ylabel('Count',   fontsize=10, color='#c8c4e0')
    ax1.tick_params(axis='x', rotation=0)
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    corr = df_clean[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdPu',
                square=True, linewidths=0.5,
                linecolor='#2e2a4a', ax=ax2,
                annot_kws={'size': 10, 'color': '#e8e6f0'})
    ax2.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', color='#e8e6f0')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('<div class="section-header">Pairplot — All Features by Species</div>', unsafe_allow_html=True)
    plot_df    = df_clean[features + ['Species']]
    colors_map = {
        'Iris-setosa'     : '#e879b0',
        'Iris-versicolor' : '#9b59f5',
        'Iris-virginica'  : '#5bbfde'
    }
    with plt.rc_context({'axes.facecolor': '#1a1730', 'figure.facecolor': '#12101e'}):
        g = sns.pairplot(plot_df, hue='Species', palette=colors_map,
                         diag_kind='hist', height=2.0, plot_kws={'alpha': 0.7})
        g.fig.suptitle('Pairplot - All Features by Species',
                        y=1.02, fontsize=12, fontweight='bold', color='#e8e6f0')
        st.pyplot(g.fig)
    plt.close()


# ================================================
# Page 3 — Outlier Handling
# ================================================
elif page == "Outlier Handling":

    st.markdown('<div class="section-header">Box Plot — Before Handling Outliers</div>', unsafe_allow_html=True)
    st.info("The small circles in the box plots below are outliers — data points that fall far outside the normal range.")

    species_list = df_original['Species'].unique()
    fig_b, axes  = plt.subplots(2, 2, figsize=(11, 8))
    fig_b.suptitle('Feature Distribution by Species (Before Handling Outliers)',
                    fontsize=12, fontweight='bold', color='#e8e6f0')

    for i, feature in enumerate(features):
        ax           = axes[i // 2, i % 2]
        data_to_plot = [df_original[df_original['Species'] == sp][feature].values for sp in species_list]
        bp           = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                                  medianprops=dict(color='#e8e6f0', linewidth=2),
                                  flierprops=dict(marker='o', color='#e879b0', markersize=5))
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(feature, fontweight='bold', fontsize=10, pad=8, color='#e8e6f0')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'], fontsize=9)
        ax.set_ylabel('cm', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    plt.tight_layout()
    st.pyplot(fig_b)
    plt.close()

    st.markdown('<div class="section-header">IQR Method — How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1** — Calculate Q1 (25th percentile) and Q3 (75th percentile) per feature per species.")
    with col2:
        st.markdown("**Step 2** — IQR = Q3 − Q1. Lower = Q1 − 1.5×IQR, Upper = Q3 + 1.5×IQR.")
    with col3:
        st.markdown("**Step 3** — Any value outside these bounds is an outlier and gets removed.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows Before", 150)
    with col2:
        st.metric("Rows After", len(df_clean), delta=f"-{outliers_removed} outliers removed")


# ================================================
# Page 4 — Model Results
# ================================================
elif page == "Model Results":

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy * 100:.2f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">100</div><div class="metric-label">Decision Trees</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">80 / 20</div><div class="metric-label">Train / Test Split</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    linewidths=0.5, linecolor='#2e2a4a',
                    annot_kws={'size': 13, 'weight': 'bold', 'color': '#e8e6f0'},
                    ax=ax_cm)
        ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='#e8e6f0')
        ax_cm.set_xlabel('Predicted Species', fontsize=10, color='#c8c4e0')
        ax_cm.set_ylabel('Actual Species',    fontsize=10, color='#c8c4e0')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        importances   = model.feature_importances_
        feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        bar_colors    = ['#5bbfde', '#5bbfde', '#e879b0', '#e879b0']
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        bars = ax_fi.barh(feature_names, importances, color=bar_colors, edgecolor='#2e2a4a')
        for bar, imp in zip(bars, importances):
            ax_fi.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                       f'{imp:.3f}', va='center', fontweight='bold', fontsize=9, color='#e8e6f0')
        ax_fi.set_title('Feature Importance', fontsize=12, fontweight='bold', color='#e8e6f0')
        ax_fi.set_xlabel('Importance Score', fontsize=10, color='#c8c4e0')
        ax_fi.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()


# ================================================
# Page 5 — Predict a Flower
# ================================================
elif page == "Predict a Flower":

    st.markdown('<div class="section-header">Enter Flower Measurements</div>', unsafe_allow_html=True)
    st.write("Adjust the sliders below and get an instant species prediction.")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.5, value=3.5, step=0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
        petal_width  = st.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=0.2, step=0.1)

    st.markdown("---")

    new_flower = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm' : [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm' : [petal_width]
    })

    prediction     = model.predict(new_flower)
    predicted_name = le.inverse_transform(prediction)[0]
    probabilities  = model.predict_proba(new_flower)[0]

    species_emoji = {
        'Iris-setosa'     : '🌸',
        'Iris-versicolor' : '🌺',
        'Iris-virginica'  : '🌼'
    }
    emoji = species_emoji.get(predicted_name, '🌸')

    st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:3.2rem; margin-bottom:0.5rem;">{emoji}</div>
            <div style="color:#6e6a85; font-size:0.85rem; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:0.4rem;">Predicted Species</div>
            <div class="prediction-species">{predicted_name}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
    for species, prob in zip(le.classes_, probabilities):
        st.write(f"**{species}**")
        st.progress(float(prob), text=f"{prob * 100:.1f}%")