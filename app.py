"""Loan Risk Predictor - Streamlit web app entry point."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from model.predict import FACTOR_LABELS, load_model, predict_risk
from model.train import BASE_FEATURES

st.set_page_config(
    page_title="Prediktor Risiko Pinjaman",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)

RISK_STYLE = {
    "Rendah": {"color": "#15803D"},
    "Sedang": {"color": "#CA8A04"},
    "Tinggi": {"color": "#B91C1C"},
}

RUPIAH_FEATURES = {"MonthlyIncome", "IncomePerDependent"}
RATIO_FEATURES = {"DebtRatio", "RevolvingUtilizationOfUnsecuredLines"}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --lp-bg: #FFFFFF;
    --lp-bg-secondary: #F8FAFC;
    --lp-border: #E2E8F0;
    --lp-text: #0F172A;
    --lp-text-muted: #64748B;
    --lp-accent: #312E81;
}

html, body, .stApp, .stApp *, [data-testid="stHeader"] *, [data-testid="stSidebar"] * {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: var(--lp-bg); }

h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    color: var(--lp-text) !important;
    letter-spacing: -0.01em;
}

.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }

[data-testid="stCaptionContainer"] p {
    color: var(--lp-text-muted) !important;
    font-size: 0.85rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--lp-bg-secondary);
    border-right: 1px solid var(--lp-border);
}
[data-testid="stSidebar"] h1 {
    font-size: 1.05rem !important;
    padding-bottom: 1.25rem;
}

/* Sidebar nav radio group styled as a tab list */
[data-testid="stRadioGroup"] { gap: 2px; }
label[data-testid="stRadioOption"] {
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 2px;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: background 0.15s ease, border-color 0.15s ease;
}
label[data-testid="stRadioOption"]:hover {
    background: rgba(49, 46, 129, 0.06);
}
label[data-testid="stRadioOption"][data-selected="true"] {
    background: rgba(49, 46, 129, 0.10);
    border-left: 3px solid var(--lp-accent);
}
label[data-testid="stRadioOption"] p {
    color: var(--lp-text) !important;
    font-weight: 500 !important;
    margin: 0;
    font-size: 0.9rem;
}
label[data-testid="stRadioOption"][data-selected="true"] p {
    color: var(--lp-accent) !important;
    font-weight: 600 !important;
}
/* hide the decorative radio dot, keep the label click target */
label[data-testid="stRadioOption"] > div > div > div:not(:has(p)) {
    display: none;
}

/* Sidebar model info card */
.lp-info-card {
    border: 1px solid var(--lp-border);
    border-radius: 10px;
    padding: 10px 14px;
    background: var(--lp-bg);
}
.lp-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
}
.lp-info-row + .lp-info-row { border-top: 1px solid var(--lp-border); }
.lp-info-label { color: var(--lp-text-muted); font-size: 0.78rem; }
.lp-info-value { color: var(--lp-text); font-size: 0.78rem; font-weight: 600; text-align: right; }

/* ---------- Card sections in main form ---------- */
/* Streamlit renders st.container(border=True) as a stLayoutWrapper whose
   stVerticalBlock's first child is an stElementContainer directly (the form's
   own outer wrapper instead nests further stLayoutWrappers first), so that
   structural difference is what distinguishes our two cards from the form shell. */
[data-testid="stForm"] [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child) {
    border: 1px solid var(--lp-border) !important;
    border-radius: 12px !important;
    background: var(--lp-bg-secondary) !important;
    box-shadow: none !important;
    padding: 1.25rem !important;
    margin-bottom: 1.25rem;
    display: block;
}
.lp-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--lp-accent);
    margin: 0 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--lp-border);
}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] label p { font-weight: 500; color: var(--lp-text); font-size: 0.88rem; }
[data-testid="stSliderThumbValue"] p { color: var(--lp-accent) !important; font-weight: 600 !important; }
[data-testid="stSliderTickBar"] p { color: var(--lp-text-muted) !important; font-size: 0.75rem; }

/* ---------- Number input ---------- */
[data-testid="stNumberInputContainer"] {
    border: 1px solid var(--lp-border) !important;
    border-radius: 8px !important;
    background: var(--lp-bg) !important;
    box-shadow: none !important;
}
[data-testid="stNumberInputField"] {
    color: var(--lp-text) !important;
    font-weight: 500;
}

/* ---------- Tooltip help icon ---------- */
[data-testid="stTooltipIcon"] button { color: var(--lp-text-muted) !important; }
[data-testid="stTooltipIcon"] button:hover { color: var(--lp-accent) !important; }
div[data-baseweb="tooltip"] { font-family: 'Poppins', sans-serif !important; }

/* ---------- Buttons ---------- */
[data-testid="stFormSubmitButton"] button {
    background: var(--lp-accent) !important;
    border: 1px solid var(--lp-accent) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    padding: 0.6rem 0 !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #26235f !important;
    border-color: #26235f !important;
}
[data-testid="stFormSubmitButton"] button p { color: #FFFFFF !important; }

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] { border: 1px solid var(--lp-border) !important; border-radius: 8px; }

/* ---------- Result card ---------- */
.lp-result-card {
    border: 1px solid var(--lp-border);
    border-radius: 12px;
    padding: 1.5rem 1.5rem 1.25rem 1.5rem;
    margin: 0 0 1.5rem 0;
    background: var(--lp-bg-secondary);
    text-align: left;
}
.lp-result-eyebrow {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--lp-text-muted);
    font-weight: 600;
    margin: 0 0 0.6rem 0;
}
.lp-result-main {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    text-align: left;
}
.lp-result-level { font-size: 1.65rem; font-weight: 700; margin: 0; }
.lp-result-pct { font-size: 1.65rem; font-weight: 700; margin: 0; }
.lp-result-subtitle { color: var(--lp-text-muted); font-size: 0.85rem; margin: 0.2rem 0 1.1rem 0; text-align: left; }
.lp-gauge-track {
    background: var(--lp-border);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.lp-gauge-fill { height: 100%; border-radius: 6px; }

.lp-factors-title {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--lp-text-muted);
    margin: 0 0 0.5rem 0;
    text-align: left;
}
.lp-factor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--lp-border);
    text-align: left;
}
.lp-factor-row:last-child { border-bottom: none; }
.lp-factor-label { display: flex; align-items: center; gap: 0.55rem; color: var(--lp-text); font-size: 0.88rem; }
.lp-factor-dot { width: 6px; height: 6px; min-width: 6px; border-radius: 50%; display: inline-block; }
.lp-factor-value { color: var(--lp-text-muted); font-size: 0.85rem; font-weight: 500; white-space: nowrap; }
.lp-no-factors { color: var(--lp-text-muted); font-size: 0.88rem; text-align: left; }
</style>
"""


def format_id(value, decimals=0):
    """Formats a number Indonesian-style: dot as thousand separator, comma as decimal."""
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(value):
    return f"Rp {format_id(value, 0)}"


def format_factor_value(feature, value):
    if feature in RUPIAH_FEATURES:
        return format_rupiah(value)
    if feature in RATIO_FEATURES:
        return format_id(value, 2)
    return format_id(value, 0)


@st.cache_resource
def get_model_bundle():
    return load_model()


def render_risk_card(result: dict):
    level = result["risk_level"]
    probability = result["probability"]
    color = RISK_STYLE[level]["color"]
    pct = round(probability * 100, 1)
    bar_pct = max(pct, 3)  # keep the fill visually meaningful even for tiny probabilities
    pct_label = format_id(pct, 1)

    st.markdown(
        f"""
        <div class="lp-result-card" style="border-color:{color}33;">
            <p class="lp-result-eyebrow">Hasil Prediksi</p>
            <div class="lp-result-main">
                <span class="lp-result-level" style="color:{color};">Risiko {level}</span>
                <span class="lp-result-pct" style="color:{color};">{pct_label}%</span>
            </div>
            <p class="lp-result-subtitle">Estimasi probabilitas gagal bayar berdasarkan profil yang dimasukkan</p>
            <div class="lp-gauge-track">
                <div class="lp-gauge-fill" style="width:{bar_pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    factors = result["top_factors"]
    factors_html = ['<p class="lp-factors-title">Faktor Utama yang Memengaruhi Prediksi</p>']
    if not factors:
        factors_html.append(
            '<p class="lp-no-factors">Tidak ditemukan faktor risiko signifikan yang meningkatkan risiko untuk pemohon ini.</p>'
        )
    else:
        for f in factors:
            value_label = format_factor_value(f["feature"], f["value"])
            factors_html.append(
                f"""
                <div class="lp-factor-row">
                    <span class="lp-factor-label"><span class="lp-factor-dot" style="background:{color};"></span>{f['label']}</span>
                    <span class="lp-factor-value">{value_label}</span>
                </div>
                """
            )
    st.markdown("".join(factors_html), unsafe_allow_html=True)


def page_predict(bundle):
    st.title("Prediktor Risiko Pinjaman")
    st.caption("Perkirakan kemungkinan pemohon pinjaman akan gagal bayar, berdasarkan profil keuangannya.")

    with st.form("predict_form"):
        with st.container(border=True):
            st.markdown('<p class="lp-card-title">Data Pribadi</p>', unsafe_allow_html=True)
            age = st.slider("Usia (tahun)", 18, 90, 35)
            monthly_income = st.number_input(
                "Penghasilan Bulanan (Rp)",
                min_value=500_000,
                max_value=100_000_000,
                value=5_000_000,
                step=100_000,
            )
            st.caption(format_rupiah(monthly_income))
            dependents = st.slider("Jumlah Tanggungan", 0, 10, 0)

        with st.container(border=True):
            st.markdown('<p class="lp-card-title">Riwayat Kredit</p>', unsafe_allow_html=True)
            open_loans = st.slider("Jumlah Pinjaman Aktif", 0, 40, 5)
            late_30_59 = st.slider(
                "Keterlambatan 30 Hari", 0, 15, 0,
                help="Jumlah keterlambatan pembayaran 30-59 hari.",
            )
            late_60_89 = st.slider(
                "Keterlambatan 60 Hari", 0, 15, 0,
                help="Jumlah keterlambatan pembayaran 60-89 hari.",
            )
            late_90 = st.slider(
                "Keterlambatan 90 Hari", 0, 15, 0,
                help="Jumlah keterlambatan pembayaran 90 hari atau lebih.",
            )
            debt_ratio = st.slider(
                "Rasio Utang", 0.0, 2.0, 0.30, step=0.01,
                help="Total pembayaran utang bulanan dibagi penghasilan bulanan.",
            )
            st.caption(f"Nilai: {format_id(debt_ratio, 2)}")
            utilization = st.slider(
                "Tingkat Penggunaan Kredit", 0.0, 1.5, 0.30, step=0.01,
                help="Saldo kartu/kredit dibandingkan dengan limit kredit.",
            )
            st.caption(f"Nilai: {format_id(utilization, 2)}")

        submitted = st.form_submit_button("Prediksi Risiko", use_container_width=True)

    if submitted:
        raw_input = {
            "age": age,
            "MonthlyIncome": float(monthly_income),
            "NumberOfDependents": dependents,
            "DebtRatio": debt_ratio,
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "NumberOfOpenCreditLinesAndLoans": open_loans,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfTimes90DaysLate": late_90,
        }
        assert set(raw_input.keys()) == set(BASE_FEATURES)
        result = predict_risk(raw_input, bundle)
        render_risk_card(result)


def page_model_info(bundle):
    st.title("Informasi Model")
    st.caption(
        f"Model terbaik yang dipilih: **{bundle['model_name']}** "
        "(dipilih berdasarkan ROC-AUC pada data uji)"
    )

    metrics_df = pd.DataFrame(bundle["metrics"]).T
    metrics_df = metrics_df[["accuracy", "roc_auc", "precision", "recall", "f1"]]
    metrics_df = metrics_df.rename(
        columns={
            "accuracy": "Akurasi",
            "roc_auc": "ROC-AUC",
            "precision": "Presisi",
            "recall": "Recall",
            "f1": "F1-Score",
        }
    )
    st.markdown("#### Perbandingan Model")
    st.dataframe(
        metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#E0E7FF"),
        use_container_width=True,
    )

    st.markdown("#### Kurva ROC")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, curve in bundle["roc_curves"].items():
        auc = bundle["metrics"][name]["roc_auc"]
        lw = 3 if name == bundle["model_name"] else 1.5
        ax.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={format_id(auc, 3)})", linewidth=lw)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Tingkat Positif Palsu")
    ax.set_ylabel("Tingkat Positif Benar")
    ax.set_title("Perbandingan Kurva ROC")
    ax.legend(loc="lower right", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    st.markdown("#### Tingkat Kepentingan Fitur")
    importance = bundle["feature_importance"]
    labels = [FACTOR_LABELS.get(f, f) for f in importance.keys()]
    values = list(importance.values())

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.barplot(x=values, y=labels, ax=ax2, color="#312E81")
    ax2.set_xlabel("Tingkat Kepentingan Relatif")
    ax2.set_ylabel("")
    ax2.set_title(f"Tingkat Kepentingan Fitur ({bundle['model_name']})")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

    st.caption(
        f"Dilatih menggunakan {format_id(bundle['test_size'])} sampel data uji "
        f"(tingkat gagal bayar dataset: {format_id(bundle['default_rate'] * 100, 1)}%), "
        f"terakhir dilatih pada {bundle['trained_at']}."
    )


def page_about():
    st.title("Tentang")
    st.markdown(
        """
Aplikasi ini memperkirakan probabilitas seorang pemohon pinjaman akan gagal
bayar, menggunakan model machine learning yang dilatih dengan data historis
peminjam. Aplikasi ini dikembangkan untuk konteks pemberian pinjaman di
Indonesia dan mengacu pada prinsip kehati-hatian yang diawasi oleh
**Otoritas Jasa Keuangan (OJK)** sebagai lembaga regulator sektor jasa
keuangan di Indonesia.

#### Dataset
Mengacu pada dataset ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit/data)
dari Kaggle, yang berisi data peminjam anonim dengan atribut keuangan serta
riwayat keterlambatan pembayaran serius dalam dua tahun. Jika berkas CSV asli
dari Kaggle tidak tersedia, alur pelatihan otomatis menggunakan dataset
sintetis dengan fitur yang sama dan hubungan risiko yang logis serta setara,
disesuaikan dengan skala penghasilan dalam Rupiah.

#### Fitur yang digunakan
- Usia
- Penghasilan bulanan (Rp)
- Jumlah tanggungan
- Rasio utang
- Tingkat penggunaan kredit
- Jumlah pinjaman aktif
- Jumlah keterlambatan pembayaran 30-59 / 60-89 / 90+ hari

#### Cara kerja
1. Data dibersihkan (nilai kosong diisi, nilai pencilan dibatasi) dan dua
   fitur tambahan direkayasa: total insiden keterlambatan dan penghasilan
   per tanggungan.
2. Tiga model — Logistic Regression, Random Forest, dan XGBoost — dilatih
   dengan pembagian data 80/20 dan dievaluasi menggunakan akurasi, ROC-AUC,
   presisi, recall, dan F1-Score.
3. Model dengan ROC-AUC terbaik disimpan dan digunakan untuk prediksi pada
   aplikasi ini.

#### Sanggahan
Aplikasi ini merupakan demo edukatif, bukan alat penilaian kredit resmi dan
tidak menggantikan proses penilaian kelayakan kredit sesuai ketentuan OJK.
Hasil prediksi tidak boleh digunakan untuk mengambil keputusan pemberian
pinjaman yang sesungguhnya.
        """
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.title("Prediktor Risiko Pinjaman")
    page = st.sidebar.radio(
        "Navigasi", ["Prediksi", "Informasi Model", "Tentang"], label_visibility="collapsed"
    )
    st.sidebar.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="lp-info-card">
            <div class="lp-info-row">
                <span class="lp-info-label">Model Aktif</span>
                <span class="lp-info-value">{bundle['model_name']}</span>
            </div>
            <div class="lp-info-row">
                <span class="lp-info-label">ROC-AUC</span>
                <span class="lp-info-value">{format_id(bundle['metrics'][bundle['model_name']]['roc_auc'], 3)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Prediksi":
        page_predict(bundle)
    elif page == "Informasi Model":
        page_model_info(bundle)
    else:
        page_about()


if __name__ == "__main__":
    main()
