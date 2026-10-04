import time
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rag.integrated_health_analysis import analyze_with_evidence
from rag.rag_llm_explainer import generate_llm_explanation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "dataset/bidmc_clean.csv"
REPLAY_SECONDS = 3
MIN_HISTORY = 2

REQUIRED_COLUMNS = [
    "patient_id",
    "recording_id",
    "time",
    "heart_rate",
    "pulse_rate",
    "spo2",
    "respiratory_rate",
]

FEATURES = [
    "heart_rate",
    "pulse_rate",
    "spo2",
    "respiratory_rate",
]

LABELS = {
    "heart_rate": "Heart Rate",
    "pulse_rate": "Pulse Rate",
    "spo2": "SpO₂",
    "respiratory_rate": "Respiratory Rate",
}

UNITS = {
    "heart_rate": "bpm",
    "pulse_rate": "/min",
    "spo2": "%",
    "respiratory_rate": "/min",
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.stApp { background: #f6f9f8; }
.main .block-container { max-width: 1450px; padding-top: 1.5rem; padding-bottom: 4rem; }
h1, h2, h3 { color: #173f3b; }
[data-testid="stSidebar"] { background: #edf6f4; border-right: 1px solid #d8e7e4; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #214943; }
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #edf8f6 100%);
    border: 1px solid #d7e9e6; border-radius: 24px; padding: 34px 38px;
    margin-bottom: 26px; box-shadow: 0 8px 30px rgba(34,83,78,.06);
}
.hero-tag { display:inline-block; color:#28786e; background:#dff1ed; padding:6px 13px;
    border-radius:999px; font-size:11px; font-weight:800; letter-spacing:.7px; margin-bottom:12px; }
.hero-title { color:#173f3b; font-size:42px; font-weight:800; line-height:1.1; margin-bottom:8px; }
.hero-subtitle { color:#718580; font-size:16px; line-height:1.55; max-width:950px; }
.hero-note { color:#52736d; font-size:12px; line-height:1.5; margin-top:14px; }
.info-card, .vital-card, .evidence-card {
    background:#fff; border:1px solid #dfeae8; border-radius:18px;
    padding:18px 20px; box-shadow:0 4px 15px rgba(35,82,77,.035);
}
.info-card { min-height:120px; }
.info-label, .vital-label { color:#788985; font-size:10px; font-weight:800; letter-spacing:.7px; }
.info-value { color:#183f3b; font-size:25px; font-weight:800; line-height:1.2; margin-top:6px; }
.info-description { color:#84928f; font-size:12px; margin-top:8px; line-height:1.5; }
.vital-value { color:#183f3b; font-size:27px; font-weight:800; margin-top:7px; }
.vital-unit { color:#8a9794; font-size:12px; }
.status-card { border-radius:18px; padding:22px 25px; margin:8px 0 25px; }
.status-normal { background:#edf8f2; border:1px solid #cae7d6; }
.status-watch { background:#fff8e9; border:1px solid #efdfb5; }
.status-attention { background:#fff1f1; border:1px solid #edcccc; }
.status-label { color:#758480; font-size:10px; font-weight:800; letter-spacing:.7px; }
.status-value { color:#214943; font-size:30px; font-weight:800; margin-top:3px; }
.status-description { color:#74847f; font-size:13px; line-height:1.55; margin-top:5px; }
.evidence-signal { color:#28786e; font-size:15px; font-weight:800; margin-bottom:5px; }
.evidence-source { color:#7c8b87; font-size:12px; margin-bottom:10px; }
.evidence-text { color:#40534f; font-size:13px; line-height:1.65; white-space:pre-wrap; }
.live-pill { color:#28786e; font-weight:800; letter-spacing:.5px; font-size:12px; }
.small-muted { color:#7b8b87; font-size:13px; }
.metric-good { color:#28786e; font-weight:800; }
.metric-watch { color:#9a741f; font-weight:800; }
.metric-alert { color:#a64a4a; font-weight:800; }
.footer { text-align:center; color:#84918e; font-size:12px; line-height:1.7; padding:25px 0 5px; }

.ai-companion{display:flex;gap:18px;align-items:center;padding:18px;border:1px solid #d7e9e5;border-radius:22px;background:linear-gradient(135deg,#fff,#f1faf7);box-shadow:0 8px 26px rgba(31,76,69,.055);margin:8px 0 14px;}
.ai-bot{position:relative;width:74px;height:74px;min-width:74px;border-radius:24px;background:#dff3ed;border:2px solid #b9ded4;display:flex;align-items:center;justify-content:center;animation:botFloat 2.8s ease-in-out infinite;}
.ai-bot-face{width:48px;height:40px;border-radius:15px;background:#fff;border:2px solid #8fc7ba;position:relative;box-shadow:0 5px 14px rgba(31,76,69,.08);}
.ai-bot-face:before{content:"•  •";position:absolute;top:5px;left:9px;color:#26786e;font-size:13px;letter-spacing:6px;}
.ai-bot-face:after{content:"⌣";position:absolute;left:17px;bottom:1px;color:#26786e;font-size:19px;font-weight:800;}
.ai-bot-orbit{position:absolute;width:88px;height:88px;border:1px dashed rgba(44,130,118,.35);border-radius:50%;animation:botOrbit 5s linear infinite;}
.ai-bot-copy{flex:1}.ai-bot-name{color:#173f3b;font-size:1rem;font-weight:850}.ai-bot-status{color:#2c8276;font-size:.65rem;font-weight:850;letter-spacing:.1em;margin-top:3px}.ai-bot-message{color:#5f7771;font-size:.78rem;line-height:1.5;margin-top:6px;}
.chat-bubble{margin-top:12px;padding:14px 16px;border-radius:17px 17px 17px 5px;background:#fff;border:1px solid #dceae6;color:#35564f;font-size:.82rem;line-height:1.65;box-shadow:0 5px 18px rgba(31,76,69,.04);}
.chat-bubble-label{color:#2c8276;font-size:.6rem;font-weight:850;letter-spacing:.1em;margin-bottom:6px;}
@keyframes botFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes botOrbit{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path):
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.copy()
    data = data.sort_values(["patient_id", "recording_id", "time"]).reset_index(drop=True)
    return data


try:
    df = load_dataset(DATA_PATH)
except Exception as exc:
    st.error("Unable to load the BIDMC dataset.")
    st.exception(exc)
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=2):
    value = safe_float(value)
    return f"{value:.{digits}f}"


def get_status_description(status):
    status = str(status).upper()
    if status == "NORMAL":
        return "No strong monitoring signal was identified by the current analysis layers."
    if status == "WATCH":
        return "At least one measurement differs from the selected personalized baseline."
    return "Stronger combined monitoring signals were identified and should be reviewed."


def status_class(status):
    status = str(status).upper()
    if status == "NORMAL":
        return "status-card status-normal"
    if status == "WATCH":
        return "status-card status-watch"
    return "status-card status-attention"


def extract_analysis(integrated):
    """Keep this tolerant because the analysis/RAG modules may return extra fields."""
    if not isinstance(integrated, dict):
        return {}, [], {}
    analysis = integrated.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}
    evidence = integrated.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    dynamic_rag = integrated.get("dynamic_rag", {})
    if not isinstance(dynamic_rag, dict):
        dynamic_rag = {}
    return analysis, evidence, dynamic_rag


def get_deviations(analysis):
    deviations = analysis.get("deviations", [])
    if isinstance(deviations, list):
        return [d for d in deviations if isinstance(d, dict)]
    return []


def get_z_scores(analysis):
    z = analysis.get("z_scores", {})
    return z if isinstance(z, dict) else {}


def get_baseline(analysis):
    baseline = analysis.get("baseline", {})
    return baseline if isinstance(baseline, dict) else {}


def get_temporal_pattern(analysis):
    value = analysis.get("temporal_pattern", "no_multisensor_pattern")
    return str(value)


def get_multisensor_events(analysis):
    events = analysis.get("multisensor_events", [])
    return events if isinstance(events, list) else []


def evidence_items(dynamic_rag, evidence):
    """Prefer the actual dynamic RAG evidence when available."""
    if isinstance(dynamic_rag, dict):
        items = dynamic_rag.get("evidence", [])
        if isinstance(items, list) and items:
            return [x for x in items if isinstance(x, dict)]
    return [x for x in evidence if isinstance(x, dict)]


def concise_evidence_text(item):
    """Turn retrieved Markdown into a compact, dashboard-friendly evidence excerpt."""
    text = item.get("text", item.get("content", ""))
    if not text:
        return "Relevant monitoring information was retrieved from the HealthSense knowledge base."

    raw = str(text).replace("\r", "").strip()
    cleaned = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line in {"---", "***", "___"}:
            continue
        # Remove Markdown heading markers so raw ## / ### never leak into the UI.
        line = line.lstrip("#").strip()
        # Keep bullets readable without exposing Markdown syntax.
        if line.startswith("- "):
            line = "• " + line[2:].strip()
        cleaned.append(line)

    # Prefer a short explanatory paragraph rather than dumping an entire KB page.
    paragraphs = []
    for line in cleaned:
        if line.startswith("• "):
            paragraphs.append(line)
        elif line.lower() in {"purpose", "oxygen saturation (spo2)", "heart rate (hr)", "pulse rate (pulse)", "respiratory rate (resp)"}:
            continue
        else:
            paragraphs.append(line)

    result = " ".join(paragraphs)
    if len(result) > 360:
        result = result[:360].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return result or "Relevant monitoring information was retrieved from the HealthSense knowledge base."


def observation_key(recording_id, index):
    return f"{recording_id}:{index}"


def llm_safe_evidence(dynamic_rag, evidence):
    """Make sure the LLM receives the structure expected by rag_llm_explainer."""
    if isinstance(dynamic_rag, dict):
        items = dynamic_rag.get("evidence", [])
        if isinstance(items, list):
            return {
                "query": dynamic_rag.get("query", ""),
                "evidence": [x for x in items if isinstance(x, dict)],
            }
    return {
        "query": "",
        "evidence": [x for x in evidence if isinstance(x, dict)],
    }


# ============================================================
# SESSION STATE
# ============================================================

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "selected_recording" not in st.session_state:
    st.session_state.selected_recording = None

if "replay_index" not in st.session_state:
    st.session_state.replay_index = 30

if "replay_recording" not in st.session_state:
    st.session_state.replay_recording = None

if "llm_explanation" not in st.session_state:
    st.session_state.llm_explanation = None

if "llm_observation_key" not in st.session_state:
    st.session_state.llm_observation_key = None

if "llm_frozen_reading" not in st.session_state:
    st.session_state.llm_frozen_reading = None

if "llm_frozen_analysis" not in st.session_state:
    st.session_state.llm_frozen_analysis = None

if "llm_frozen_evidence" not in st.session_state:
    st.session_state.llm_frozen_evidence = None


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-tag">REAL-DATA INTELLIGENCE · BIDMC PHYSIOLOGICAL MONITORING</div>
    <div class="hero-title">HealthSense AI</div>
    <div class="hero-subtitle">Personalized IoT Healthcare Monitoring & Early-Warning Assistant</div>
    <div class="hero-note">
        Real BIDMC physiological data • Personalized baseline • Temporal analysis •
        Isolation Forest anomaly signal • RAG evidence • Llama 3.2 explanation
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TOP DATASET SUMMARY
# ============================================================

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="info-card"><div class="info-label">OBSERVATIONS</div>'
        f'<div class="info-value">{len(df):,}</div>'
        f'<div class="info-description">Real physiological observations</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="info-card"><div class="info-label">BIDMC RECORDS</div>'
        f'<div class="info-value">{df["patient_id"].nunique()}</div>'
        f'<div class="info-description">Real BIDMC recordings represented</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="info-card"><div class="info-label">RECORDINGS</div>'
        f'<div class="info-value">{df["recording_id"].nunique()}</div>'
        f'<div class="info-description">PhysioNet BIDMC physiological dataset</div></div>',
        unsafe_allow_html=True,
    )

st.caption("Academic/research monitoring prototype. Not a clinical diagnostic system.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🩺 Monitoring Controls")
st.sidebar.caption("Select a real BIDMC patient and recording.")

patients = sorted(df["patient_id"].dropna().unique().tolist())

patient_default = (
    st.session_state.selected_patient
    if st.session_state.selected_patient in patients
    else patients[0]
)

selected_patient = st.sidebar.selectbox(
    "Patient",
    patients,
    index=patients.index(patient_default),
)

patient_df = df[df["patient_id"] == selected_patient].copy()
recordings = sorted(patient_df["recording_id"].dropna().unique().tolist())

recording_default = (
    st.session_state.selected_recording
    if st.session_state.selected_recording in recordings
    else recordings[0]
)

selected_recording = st.sidebar.selectbox(
    "Recording",
    recordings,
    index=recordings.index(recording_default),
)

selection_changed = (
    selected_patient != st.session_state.selected_patient
    or selected_recording != st.session_state.selected_recording
)

if selection_changed:
    st.session_state.selected_patient = selected_patient
    st.session_state.selected_recording = selected_recording
    st.session_state.replay_recording = selected_recording
    st.session_state.replay_index = min(30, max(1, len(patient_df[patient_df["recording_id"] == selected_recording]) - 1))
    st.session_state.llm_explanation = None
    st.session_state.llm_observation_key = None
    st.session_state.llm_frozen_reading = None
    st.session_state.llm_frozen_analysis = None
    st.session_state.llm_frozen_evidence = None

recording_df = patient_df[patient_df["recording_id"] == selected_recording].copy()
recording_df = recording_df.sort_values("time").reset_index(drop=True)

if len(recording_df) < MIN_HISTORY:
    st.error("This recording does not contain enough observations for monitoring analysis.")
    st.stop()

if st.session_state.replay_recording != selected_recording:
    st.session_state.replay_recording = selected_recording
    st.session_state.replay_index = min(30, len(recording_df) - 1)

st.sidebar.divider()
st.sidebar.markdown("### Replay")
st.sidebar.caption(f"The dashboard automatically advances every {REPLAY_SECONDS} seconds.")

manual_start = st.sidebar.slider(
    "Start / jump to observation",
    min_value=1,
    max_value=len(recording_df) - 1,
    value=min(int(st.session_state.replay_index), len(recording_df) - 1),
)

if manual_start != st.session_state.replay_index:
    st.session_state.replay_index = manual_start
    st.session_state.llm_explanation = None
    st.session_state.llm_observation_key = None
    st.session_state.llm_frozen_reading = None
    st.session_state.llm_frozen_analysis = None
    st.session_state.llm_frozen_evidence = None

st.sidebar.caption("Move the slider to jump; otherwise the replay continues automatically.")
st.sidebar.info("🧠 Mira / Llama 3.2: start Ollama only when you click the AI explanation button.")
st.sidebar.divider()
st.sidebar.caption("Dataset: PhysioNet BIDMC PPG and Respiration Dataset")
st.sidebar.caption("Monitoring prototype — not a clinical diagnostic system.")


# ============================================================
# SYNCHRONIZED LIVE PIPELINE
# ============================================================

@st.fragment(run_every=REPLAY_SECONDS)
def synchronized_live_pipeline():
    """
    Every automatic fragment rerun calculates the current observation first.
    All dynamic sections below use exactly that same index and analysis result.
    """

    total_rows = len(recording_df)

    current_index = int(st.session_state.replay_index)
    current_index = max(1, min(current_index, total_rows - 1))
    st.session_state.replay_index = current_index

    current_row = recording_df.iloc[current_index]
    analysis_rows = recording_df.iloc[: current_index + 1].copy()

    # --------------------------------------------------------
    # HEALTHSENSE ANALYSIS FOR THIS EXACT OBSERVATION
    # --------------------------------------------------------

    try:
        with st.spinner("Updating HealthSense analysis…"):
            integrated_result = analyze_with_evidence(analysis_rows)
    except Exception as exc:
        st.error("Integrated HealthSense analysis failed.")
        st.exception(exc)
        return

    analysis, evidence, dynamic_rag = extract_analysis(integrated_result)

    # --------------------------------------------------------
    # CURRENT VALUES
    # --------------------------------------------------------

    current_values = {
        "heart_rate": safe_float(current_row["heart_rate"]),
        "pulse_rate": safe_float(current_row["pulse_rate"]),
        "spo2": safe_float(current_row["spo2"]),
        "respiratory_rate": safe_float(current_row["respiratory_rate"]),
    }

    current_time = safe_float(current_row["time"])
    current_key = observation_key(selected_recording, current_index)

    status = str(analysis.get("status", "WATCH")).upper()
    deviations = get_deviations(analysis)
    z_scores = get_z_scores(analysis)
    baseline = get_baseline(analysis)
    temporal_pattern = get_temporal_pattern(analysis)
    multisensor_events = get_multisensor_events(analysis)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.subheader("● LIVE REPLAY")
    st.markdown(
        f'<div class="live-pill">AUTOMATICALLY SYNCHRONIZED · OBSERVATION {current_index + 1} / {total_rows}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Dataset time: {current_time:.0f}s · Patient: {selected_patient} · Recording: {selected_recording}"
    )

    # --------------------------------------------------------
    # OVERVIEW + STATUS
    # --------------------------------------------------------

    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(
            f'<div class="info-card"><div class="info-label">PATIENT</div>'
            f'<div class="info-value">{escape(str(selected_patient))}</div>'
            f'<div class="info-description">Real BIDMC physiological recording</div></div>',
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            f'<div class="info-card"><div class="info-label">RECORDING</div>'
            f'<div class="info-value">{escape(str(selected_recording))}</div>'
            f'<div class="info-description">{total_rows} observations available</div></div>',
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            f'<div class="info-card"><div class="info-label">ANALYSIS SNAPSHOT</div>'
            f'<div class="info-value">{current_index + 1} / {total_rows}</div>'
            f'<div class="info-description">Exactly the live observation shown below</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="{status_class(status)}">'
        f'<div class="status-label">CURRENT MONITORING STATUS</div>'
        f'<div class="status-value">{escape(status)}</div>'
        f'<div class="status-description">{escape(get_status_description(status))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LIVE VITALS
    # --------------------------------------------------------

    st.subheader("🩺 Live Physiological Readings")
    m1, m2, m3, m4 = st.columns(4)
    for col, feature in zip((m1, m2, m3, m4), FEATURES):
        with col:
            value = current_values[feature]
            unit = UNITS[feature]
            label = LABELS[feature]
            st.markdown(
                f'<div class="vital-card"><div class="vital-label">{label.upper()}</div>'
                f'<div class="vital-value">{value:.0f} <span class="vital-unit">{unit}</span></div></div>',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # LIVE CHARTS
    # --------------------------------------------------------

    WINDOW = 90
    start_idx = max(0, current_index - WINDOW + 1)
    chart_df = recording_df.iloc[start_idx: current_index + 1].copy()

    st.subheader("📈 Living Physiological Stream")
    st.caption("The graph reveals the real BIDMC recording as the replay advances. The live marker, analysis and fingerprint all use the same observation.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["heart_rate"], mode="lines", name="Heart Rate", line=dict(width=2.7, color="#2c8276"), hovertemplate="Heart Rate · %{y:.0f} bpm<br>Time · %{x:.0f}s<extra></extra>"))
    fig.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["pulse_rate"], mode="lines", name="Pulse Rate", line=dict(width=2.2, color="#8cb9af"), hovertemplate="Pulse Rate · %{y:.0f}/min<br>Time · %{x:.0f}s<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current_time], y=[current_values["heart_rate"]], mode="markers", name="LIVE", marker=dict(size=15, color="#173f3b", line=dict(width=4, color="white")), hovertemplate="<b>LIVE OBSERVATION</b><br>%{x:.0f}s · HR %{y:.0f} bpm<extra></extra>"))
    fig.add_vline(x=current_time, line_dash="dot", line_width=2, line_color="#7daea4")
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=35,b=30), hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white", xaxis_title="Live recording time", yaxis_title="Rate / minute", legend=dict(orientation="h",y=1.08,x=0), xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e7efec"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False,"responsive":True})

    left,right=st.columns(2)
    with left:
        fig_spo2=go.Figure()
        fig_spo2.add_trace(go.Scatter(x=chart_df["time"],y=chart_df["spo2"],mode="lines",name="SpO₂",line=dict(width=2.5,color="#7b61a8")))
        fig_spo2.add_trace(go.Scatter(x=[current_time],y=[current_values["spo2"]],mode="markers",name="LIVE",marker=dict(size=12,color="#7b61a8",line=dict(width=3,color="white"))))
        fig_spo2.add_vline(x=current_time,line_dash="dot",line_width=1.5)
        fig_spo2.update_layout(height=300,margin=dict(l=20,r=20,t=25,b=25),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Live time",yaxis_title="SpO₂ (%)",xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#e7efec"),showlegend=False)
        st.plotly_chart(fig_spo2,use_container_width=True,config={"displayModeBar":False,"responsive":True})
    with right:
        fig_resp=go.Figure()
        fig_resp.add_trace(go.Scatter(x=chart_df["time"],y=chart_df["respiratory_rate"],mode="lines",name="Respiratory Rate",line=dict(width=2.5,color="#b07d27")))
        fig_resp.add_trace(go.Scatter(x=[current_time],y=[current_values["respiratory_rate"]],mode="markers",name="LIVE",marker=dict(size=12,color="#b07d27",line=dict(width=3,color="white"))))
        fig_resp.add_vline(x=current_time,line_dash="dot",line_width=1.5)
        fig_resp.update_layout(height=300,margin=dict(l=20,r=20,t=25,b=25),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Live time",yaxis_title="Respiratory rate / min",xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#e7efec"),showlegend=False)
        st.plotly_chart(fig_resp,use_container_width=True,config={"displayModeBar":False,"responsive":True})

    # --------------------------------------------------------
    # WHAT CHANGED
    # --------------------------------------------------------

    if current_index > 0:
        previous = recording_df.iloc[current_index - 1]
        st.subheader("⚡ What changed from the previous observation?")
        change_cols = st.columns(4)
        for col, feature in zip(change_cols, FEATURES):
            delta = current_values[feature] - safe_float(previous[feature])
            with col:
                st.metric(LABELS[feature], f"{current_values[feature]:.2f}", f"{delta:+.2f}")

    # --------------------------------------------------------
    # LIVE FINGERPRINT
    # --------------------------------------------------------

    st.subheader("🧬 Live Physiological Fingerprint")
    st.caption("A four-sensor shape representing the current observation relative to its personalized baseline.")
    fingerprint_labels=[LABELS[f] for f in FEATURES]
    z_values=[safe_float(z_scores.get(f,0.0)) for f in FEATURES]
    clipped=[max(-3.0,min(3.0,z)) for z in z_values]
    radar_values=[z+3.0 for z in clipped]
    radar_fig=go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=radar_values+[radar_values[0]],theta=fingerprint_labels+[fingerprint_labels[0]],mode="lines+markers",fill="toself",name="Live fingerprint",line=dict(width=3,color="#2c8276"),marker=dict(size=8,color="#173f3b"),customdata=z_values+[z_values[0]],hovertemplate="%{theta}<br>Personalized z-score · %{customdata:+.2f}σ<extra></extra>"))
    radar_fig.update_layout(height=400,margin=dict(l=35,r=35,t=25,b=25),paper_bgcolor="white",showlegend=False,polar=dict(bgcolor="white",radialaxis=dict(visible=False,range=[0,6]),angularaxis=dict(gridcolor="#dfeae6",linecolor="#dfeae6")))
    fp_left,fp_right=st.columns([1.35,1])
    with fp_left:
        st.plotly_chart(radar_fig,use_container_width=True,config={"displayModeBar":False,"responsive":True})
    with fp_right:
        for feature,z in zip(FEATURES,z_values):
            st.metric(LABELS[feature],f"{z:+.2f}σ")
            st.caption("outside ±2σ" if abs(z)>=2 else "within ±2σ")

    # --------------------------------------------------------
    # --------------------------------------------------------
    # PERSONALIZED BASELINE
    # --------------------------------------------------------

    st.subheader("🎯 Personalized Baseline")
    st.caption(f"Each card compares the live reading with the recording-specific history available before observation {current_index + 1}.")

    base_cols = st.columns(4)
    for col, feature in zip(base_cols, FEATURES):
        with col:
            info = baseline.get(feature, {}) if isinstance(baseline.get(feature, {}), dict) else {}
            mean = info.get("mean", info.get("baseline_mean", None))
            std = info.get("std", info.get("baseline_std", None))
            count = info.get("count", max(0, current_index))
            value = current_values[feature]
            mean_value = safe_float(mean) if mean is not None else None
            spread = safe_float(std) if std is not None else 0.0
            delta = value - mean_value if mean_value is not None else 0.0
            label = escape(LABELS[feature])
            unit = escape(UNITS[feature])
            current_text = f"{value:.2f} {unit}"
            delta_class = "baseline-up" if delta > 0 else "baseline-down" if delta < 0 else "baseline-flat"
            delta_text = f"{delta:+.2f}"
            if mean_value is not None:
                detail = f"Baseline {mean_value:.2f} · spread ±{spread:.2f}"
            else:
                detail = "Baseline information unavailable"
            st.html(f"""
            <div class=\"baseline-card\">
                <div class=\"baseline-card-label\">{label}</div>
                <div class=\"baseline-current\">{current_text}</div>
                <div class=\"baseline-delta {delta_class}\">{delta_text} from baseline</div>
                <div class=\"baseline-detail\">{detail}</div>
                <div class=\"baseline-count\">{int(count)} historical observations</div>
            </div>
            """)

    # --------------------------------------------------------
    # PERSONALIZED DEVIATION
    # --------------------------------------------------------

    st.subheader("📊 Personalized Deviation")
    st.caption("The z-score shows how far the live reading is from this recording's personalized baseline.")

    dev_cols = st.columns(4)
    for col, feature in zip(dev_cols, FEATURES):
        with col:
            z = safe_float(z_scores.get(feature, 0.0))
            label = escape(LABELS[feature])
            outside = abs(z) >= 2
            state = "Outside ±2σ reference" if outside else "Within ±2σ reference"
            state_class = "deviation-alert" if outside else "deviation-normal"
            st.html(f"""
            <div class=\"deviation-card\">
                <div class=\"deviation-label\">{label}</div>
                <div class=\"deviation-value\">{z:+.2f}σ</div>
                <div class=\"deviation-state {state_class}\">{state}</div>
            </div>
            """)

    # --------------------------------------------------------
    # ML SIGNAL
    # --------------------------------------------------------

    st.subheader("🤖 AI Monitoring Signals")
    st.caption("The ML detector is a supporting anomaly signal; personalized baseline analysis remains separate.")

    ml_result = str(analysis.get("ml_result", analysis.get("ml_status", "UNKNOWN"))).upper()
    ml_score = analysis.get("ml_score", analysis.get("anomaly_score", None))
    signal_count = analysis.get("signal_count", len(deviations))

    ml1, ml2, ml3, ml4 = st.columns(4)
    ml_cards = [
        ("Overall monitoring", status, "Personalized + temporal signals", "status"),
        ("Isolation Forest", ml_result, "Supporting ML anomaly signal", "ml"),
        ("Anomaly score", f"{safe_float(ml_score):.5f}" if ml_score is not None else "N/A", "Current model output", "score"),
        ("Monitoring signals", str(signal_count), "Baseline / temporal / multisensor", "count"),
    ]
    for col, (title, value, detail, kind) in zip((ml1, ml2, ml3, ml4), ml_cards):
        with col:
            value_class = "signal-attention" if str(value).upper() in {"ATTENTION", "WATCH", "ANOMALOUS"} else "signal-stable"
            st.html(f"""
            <div class=\"ai-signal-card\">
                <div class=\"ai-signal-label\">{escape(title)}</div>
                <div class=\"ai-signal-value {value_class}\">{escape(str(value))}</div>
                <div class=\"ai-signal-detail\">{escape(detail)}</div>
            </div>
            """)

    if deviations:
        st.markdown("### ⚠ Monitoring Signals")
        signal_cols = st.columns(min(3, len(deviations)))
        for idx, d in enumerate(deviations):
            label = d.get("label", LABELS.get(d.get("sensor"), d.get("sensor", "Measurement")))
            value = safe_float(d.get("value", current_values.get(d.get("sensor"), 0)))
            z = safe_float(d.get("z_score", 0))
            direction = d.get("direction", "different from baseline")
            with signal_cols[idx % len(signal_cols)]:
                st.html(f"""<div class=\"signal-detail-card\"><b>{escape(str(label))}</b><span>Current {value:.2f}</span><span>z = {z:+.2f}</span><span>{escape(str(direction))}</span></div>""")
    else:
        st.success("No personalized baseline deviation was detected in the current observation.")

    # --------------------------------------------------------
    # TEMPORAL + MULTISENSOR
    # --------------------------------------------------------

    st.subheader("🔗 Temporal & Multisensor Analysis")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Temporal Pattern**")
        st.write(temporal_pattern)
    with t2:
        st.markdown("**Multisensor Events**")
        st.write(len(multisensor_events))

    # --------------------------------------------------------
    # RAG EVIDENCE
    # --------------------------------------------------------

    st.subheader("📚 Evidence-Supported Monitoring")
    st.caption("Evidence is retrieved for the same live observation used by the analysis above.")

    items = evidence_items(dynamic_rag, evidence)
    if items:
        evidence_cols = st.columns(min(3, len(items[:3])))
        for card_index, item in enumerate(items[:3]):
            source = item.get("source", item.get("metadata", {}).get("source", "Knowledge Base"))
            signal = item.get("signal", item.get("topic", "Monitoring Evidence"))
            with evidence_cols[card_index % len(evidence_cols)]:
                st.markdown(
                    f'<div class="evidence-card">'
                    f'<div class="evidence-signal">🔎 {escape(str(signal))}</div>'
                    f'<div class="evidence-source">Source · {escape(str(source))}</div>'
                    f'<div class="evidence-text">{escape(concise_evidence_text(item))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No evidence items were returned for this observation.")

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader("🧠 Monitoring Interpretation")
    if deviations:
        first = deviations[0]
        label = first.get("label", "measurement")
        direction = first.get("direction", "different from baseline")
        st.info(
            f"The current {label} measurement is {direction} the selected recording's personalized baseline. "
            "This is a monitoring signal and should be interpreted with the surrounding observation history."
        )
    else:
        st.info(
            "The current observation does not show a strong personalized-baseline deviation. "
            "The result should still be interpreted with the surrounding observation history."
        )

    # --------------------------------------------------------
    # LLAMA — SAME CURRENT OBSERVATION
    # --------------------------------------------------------

    st.subheader("🤖 Mira · HealthSense AI Companion")
    st.html(f"""<div class="ai-companion"><div class="ai-bot"><div class="ai-bot-orbit"></div><div class="ai-bot-face"></div></div><div class="ai-bot-copy"><div class="ai-bot-name">Mira</div><div class="ai-bot-status">GROUNDED LOCAL AI · LLAMA 3.2</div><div class="ai-bot-message">Hi! I can explain this exact observation using the HealthSense analysis and retrieved evidence. I do not diagnose or recommend treatment.</div></div></div>""")
    st.caption(f"Mira receives observation {current_index + 1} / {total_rows} · dataset time {current_time:.0f}s · the same live values shown above.")

    # Keep the button identity stable while the live replay moves.
    llm_button = st.button(
        "✨ Ask Mira to explain this observation",
        type="primary",
        key="mira_explanation_button",
        use_container_width=True,
    )

    if llm_button:
        # Freeze the exact observation shown when the user clicked Mira.
        frozen_reading = {
            "patient_id": selected_patient,
            "recording_id": selected_recording,
            "time": current_time,
            **current_values,
        }

        frozen_evidence = llm_safe_evidence(dynamic_rag, evidence)

        st.session_state.llm_frozen_reading = frozen_reading
        st.session_state.llm_frozen_analysis = analysis
        st.session_state.llm_frozen_evidence = frozen_evidence

        try:
            with st.spinner("Llama 3.2 is generating a grounded explanation…"):
                llm_result = generate_llm_explanation(
                    analysis,
                    frozen_reading,
                    evidence=frozen_evidence,
                )

            if isinstance(llm_result, dict):
                explanation = str(llm_result.get("explanation", "")).strip()
            else:
                explanation = str(llm_result).strip()

            if explanation:
                st.session_state.llm_explanation = explanation
                st.session_state.llm_observation_key = (
                    frozen_reading["patient_id"],
                    frozen_reading["recording_id"],
                    frozen_reading["time"],
                )
            else:
                st.warning("Llama returned an empty explanation.")

        except Exception as exc:
            st.error("Llama explanation generation failed.")
            st.exception(exc)

    # Mira's explanation belongs to the frozen snapshot, not the moving
    # live observation.
    if st.session_state.llm_explanation:
        frozen_reading = st.session_state.llm_frozen_reading

        st.html(
            f"""<div class="chat-bubble">
            <div class="chat-bubble-label">MIRA · GROUNDED EXPLANATION</div>
            {escape(str(st.session_state.llm_explanation))}
            </div>"""
        )

        if frozen_reading:
            st.caption(
                f"Explanation generated for observation at "
                f"{safe_float(frozen_reading['time']):.0f}s "
                f"({escape(str(frozen_reading['recording_id']))}). "
                f"It uses the retrieved evidence for that exact observation."
            )
    else:
        st.caption(
            "Generate the explanation when you want Llama to interpret this "
            "exact live observation. The replay does not call Llama automatically."
        )

    # --------------------------------------------------------
    # DECISION PIPELINE
    # --------------------------------------------------------

    st.subheader("🔬 Decision Pipeline")
    pipeline = [
        ("01", "Real BIDMC Data", "Current physiological observation"),
        ("02", "Personal Baseline", "Historical comparison"),
        ("03", "Deviation Analysis", "Personalized z-scores"),
        ("04", "ML Signal", "Isolation Forest"),
        ("05", "Evidence Retrieval", "RAG knowledge support"),
        ("06", "Human Review", "Monitoring interpretation"),
    ]
    for number, title, detail in pipeline:
        p1, p2 = st.columns([1, 5])
        with p1:
            st.markdown(f"### {number}")
        with p2:
            st.markdown(f"**{title}**")
            st.caption(detail)

    # --------------------------------------------------------
    # HUMAN IN THE LOOP
    # --------------------------------------------------------

    st.subheader("👤 Human-in-the-Loop Review")
    st.info(
        "HealthSense AI combines personalized baseline deviation, temporal/multisensor patterns, "
        "ML anomaly signals and retrieved evidence to support human review. An anomaly signal does "
        "not automatically indicate a medical condition, and the prototype does not make autonomous clinical decisions."
    )

    # --------------------------------------------------------
    # NEXT OBSERVATION
    # --------------------------------------------------------

    if current_index < total_rows - 1:
        st.session_state.replay_index = current_index + 1
    else:
        # Restart automatically from the first analyzable observation.
        st.session_state.replay_index = 1
        st.session_state.llm_explanation = None
        st.session_state.llm_observation_key = None
        st.session_state.llm_frozen_reading = None
        st.session_state.llm_frozen_analysis = None
        st.session_state.llm_frozen_evidence = None
        st.caption("Replay reached the end of the recording and will restart from observation 2.")


synchronized_live_pipeline()


# ============================================================
# STATIC PROJECT INFORMATION
# ============================================================

st.divider()
st.subheader("📊 Dataset & System Architecture")

left, right = st.columns(2)
with left:
    st.markdown(
        """
### PhysioNet BIDMC dataset

The project uses the real **BIDMC PPG and Respiration Dataset**.

Tracked measurements:

- Heart Rate
- Pulse Rate
- SpO₂
- Respiratory Rate

The dashboard replays the real observations rather than generating synthetic physiological values for the analysis layer.
"""
    )

with right:
    st.markdown(
        """
### System architecture

```text
Real BIDMC Dataset
        ↓
Preprocessing
        ↓
Real-data replay
        ↓
Personalized Baseline
        ↓
Temporal + Multisensor Analysis
        ↓
Isolation Forest
        ↓
RAG Evidence Retrieval
        ↓
Llama 3.2
        ↓
Streamlit Dashboard
        ↓
Human Review
```

This is an engineering/research prototype for personalized monitoring and early-warning support.
"""
    )


with st.expander("How is the monitoring status calculated?"):
    st.markdown(
        """
1. **Personalized baseline** — previous observations from the selected recording provide the comparison history.
2. **Deviation analysis** — the current measurements are compared with that history using z-scores.
3. **Temporal/multisensor analysis** — recent observations are examined for patterns across the available physiological measurements.
4. **Isolation Forest** — the pretrained real-BIDMC model provides a supporting anomaly signal.
5. **Evidence retrieval** — relevant knowledge-base content is retrieved for the same observation.
6. **Llama 3.2** — when requested, Llama explains the same current observation using the HealthSense analysis and retrieved evidence.
7. **Human review** — the output is monitoring support, not an autonomous clinical decision.
        """
    )


st.markdown(
    """
<div class="footer">
<strong>HealthSense AI</strong><br>
Personalized IoT Healthcare Monitoring & Early-Warning Assistant<br>
PhysioNet BIDMC physiological dataset · Academic/research prototype · Not a clinical diagnostic system.
</div>
""",
    unsafe_allow_html=True,
)
