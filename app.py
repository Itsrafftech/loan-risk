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
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

RISK_STYLE = {
    "Rendah": {"color": "#16a34a", "bg": "#dcfce7", "icon": "✅"},
    "Sedang": {"color": "#d97706", "bg": "#fef3c7", "icon": "⚠️"},
    "Tinggi": {"color": "#dc2626", "bg": "#fee2e2", "icon": "🚨"},
}

RUPIAH_FEATURES = {"MonthlyIncome", "IncomePerDependent"}
RATIO_FEATURES = {"DebtRatio", "RevolvingUtilizationOfUnsecuredLines"}

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 800px; }
    div[data-testid="stMetric"] {
        background: rgba(127,127,127,0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .risk-card {
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1.25rem 0;
    }
    .risk-card h2 { margin: 0 0 0.25rem 0; }
    .risk-card p { margin: 0; opacity: 0.85; }
    .factor-list { margin-top: 0.5rem; }
    .prog-track {
        background: rgba(127,127,127,0.18);
        border-radius: 8px;
        height: 24px;
        width: 100%;
        overflow: hidden;
    }
    .prog-fill {
        height: 100%;
        border-radius: 8px;
        text-align: right;
        color: white;
        font-size: 12px;
        font-weight: 600;
        line-height: 24px;
        padding-right: 10px;
        white-space: nowrap;
    }
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
    style = RISK_STYLE[level]
    pct = round(probability * 100, 1)
    # Keep the fill visually meaningful even for very small probabilities.
    bar_pct = max(pct, 3)
    pct_label = format_id(pct, 1)

    st.markdown(
        f"""
        <div class="risk-card" style="background:{style['bg']}; border: 1px solid {style['color']}44;">
            <h2 style="color:{style['color']};">{style['icon']} Risiko {level}</h2>
            <p style="color:{style['color']};">Estimasi probabilitas gagal bayar: <strong>{pct_label}%</strong></p>
        </div>
        <div class="prog-track">
            <div class="prog-fill" style="width:{bar_pct}%; background:{style['color']};">{pct_label}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Faktor utama yang memengaruhi prediksi ini")
    factors = result["top_factors"]
    if not factors:
        st.markdown("Tidak ditemukan faktor risiko signifikan yang meningkatkan risiko untuk pemohon ini.")
    else:
        lines = "\n".join(
            f"- **{f['label']}**: {format_factor_value(f['feature'], f['value'])}" for f in factors
        )
        st.markdown(lines)


def page_predict(bundle):
    st.title("💳 Prediktor Risiko Pinjaman")
    st.caption("Perkirakan kemungkinan pemohon pinjaman akan gagal bayar, berdasarkan profil keuangannya.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
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
        with col2:
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
    st.title("📊 Informasi Model")
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
        metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#dcfce7"),
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
    sns.barplot(x=values, y=labels, ax=ax2, color="#2563eb")
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
    st.title("ℹ️ Tentang")
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
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model aktif: **{bundle['model_name']}**")
    st.sidebar.caption(f"ROC-AUC: {format_id(bundle['metrics'][bundle['model_name']]['roc_auc'], 3)}")

    if page == "Prediksi":
        page_predict(bundle)
    elif page == "Informasi Model":
        page_model_info(bundle)
    else:
        page_about()


if __name__ == "__main__":
    main()
