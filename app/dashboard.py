import os
from types import SimpleNamespace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL",
                              "http://localhost:8000").rstrip("/")
API_TIMEOUT = 15

st.set_page_config(page_title="Amplicast · Weather Intelligence",
                   page_icon=":cyclone:", layout="wide",
                   initial_sidebar_state="expanded")

SEV_COLOR = {"low": "#00E5C7", "moderate": "#FFB020", "severe": "#FF4D4D"}
SEV_RANK = {"low": 1, "moderate": 2, "severe": 3}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root{
  --dc-sans:'Space Grotesk','Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --dc-mono:'JetBrains Mono','Space Mono',ui-monospace,'SF Mono',Consolas,monospace;
  --dc-text:#E6EDF6; --dc-muted:#8FA1B7; --dc-faint:#61748C;
  --dc-teal:#00E5C7; --dc-amber:#FFB020; --dc-red:#FF4D4D;
  --dc-hair:rgba(0,229,199,.25);
}

html, body, .stApp{
  background:radial-gradient(circle at top,#0d1626,#060a12) !important;
  background-color:#060a12 !important;
  color:var(--dc-text);
  font-family:var(--dc-sans);
}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(circle at top,#0d1626,#060a12) !important;
  background-color:#060a12 !important;
  color:var(--dc-text);
}
[data-testid="stSidebar"]{
  background:radial-gradient(circle at top,#0f1a2b,#060a12) !important;
  background-color:#060a12 !important;
  border-right:1px solid var(--dc-hair);
  box-shadow:1px 0 0 rgba(0,0,0,.6);
}
[data-testid="stSidebar"][aria-expanded="true"]{
  width:340px !important;
  min-width:300px !important;
  max-width:340px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  background:transparent;
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{
  padding-bottom:56px;
}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-live){
  position:static !important;
}

.sidebar-brand{
  font-family:var(--dc-mono);
  font-weight:700;
  font-size:1.1rem;
  line-height:1.3;
  letter-spacing:2px;
  text-transform:uppercase;
  color:#fff;
  text-shadow:0 0 18px rgba(0,229,199,.6);
  border-bottom:2px solid rgba(0,229,199,.55);
  box-shadow:0 3px 16px -6px rgba(0,229,199,.85);
  padding-bottom:7px;
  margin:2px 0 0;
}
.sidebar-brandsub{
  font-family:var(--dc-mono);
  font-size:.6rem;
  line-height:1.4;
  letter-spacing:2px;
  text-transform:uppercase;
  color:var(--dc-faint);
  margin:7px 0 4px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label{
  font-family:var(--dc-sans);
  font-size:1.05rem;
  letter-spacing:.02em;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input){
  display:flex;
  align-items:center;
  padding:11px 12px;
  margin:2px 0;
  border-radius:9px;
  border-left:3px solid transparent;
  color:var(--dc-muted);
  cursor:pointer;
  transition:background .18s ease, border-color .18s ease, color .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input):hover{
  background:rgba(255,255,255,.05);
  color:var(--dc-text);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){
  background:rgba(0,229,199,.08);
  border-left-color:var(--dc-teal);
  color:#fff;
  box-shadow:inset 0 0 26px -12px rgba(0,229,199,.95);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span{
  color:#fff;
  text-shadow:0 0 14px rgba(0,229,199,.55);
}
@media (prefers-reduced-motion:reduce){
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input){
    transition:none;
  }
}

.sidebar-live{
  position:absolute;
  left:20px;
  bottom:18px;
  display:flex;
  align-items:center;
  gap:9px;
  width:max-content;
  padding:7px 14px;
  border:1px solid rgba(0,229,199,.35);
  border-radius:999px;
  background:rgba(0,229,199,.07);
  font-family:var(--dc-mono);
  font-size:.64rem;
  font-weight:600;
  letter-spacing:.24em;
  text-transform:uppercase;
  color:var(--dc-teal);
  box-shadow:0 0 22px -8px rgba(0,229,199,.95);
}
.sidebar-live .dot{
  width:8px;
  height:8px;
  border-radius:50%;
  background:#3BFFC9;
  box-shadow:0 0 10px rgba(0,229,199,1);
  animation:dc-dot-pulse 1.5s ease-in-out infinite;
}
@keyframes dc-dot-pulse{
  0%,100%{opacity:1; transform:scale(1);}
  50%{opacity:.3; transform:scale(.72);}
}
@media (prefers-reduced-motion:reduce){
  .sidebar-live .dot{animation:none;}
}
[data-testid="stHeader"], [data-testid="stToolbar"]{background:transparent;}
[data-testid="stDecoration"], [data-testid="stStatusWidget"]{opacity:.55;}

h1,h2,h3,h4,h5{
  font-family:var(--dc-sans); color:#fff;
  letter-spacing:.01em; font-weight:600;
}
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] small{
  font-family:var(--dc-mono); font-size:.72rem; letter-spacing:.07em;
  color:var(--dc-muted);
}
[data-testid="stMarkdownContainer"] strong{font-family:var(--dc-mono); color:#fff;}
code, kbd, pre{
  font-family:var(--dc-mono) !important;
  background:rgba(0,229,199,.08) !important; color:#9FF0E2 !important;
  border:1px solid var(--dc-hair); border-radius:5px; padding:1px 5px;
}
hr, [data-testid="stDivider"]{border-color:rgba(130,168,200,.16) !important;}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-track{background:rgba(255,255,255,.02);}
::-webkit-scrollbar-thumb{background:rgba(0,229,199,.28);border-radius:9px;}
::-webkit-scrollbar-thumb:hover{background:rgba(0,229,199,.5);}

[data-testid="stMetric"]{
  background:linear-gradient(158deg, rgba(15,22,36,.60), rgba(11,17,29,.42));
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(0,229,199,.25);
  border-radius:14px;
  padding:14px 18px;
  box-shadow:0 16px 40px -28px rgba(0,0,0,.95),
             inset 0 1px 0 rgba(255,255,255,.05),
             0 0 22px -14px rgba(0,229,199,.7);
  transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
[data-testid="stMetric"]:hover{
  transform:translateY(-2px);
  border-color:rgba(0,229,199,.5);
  box-shadow:0 22px 48px -26px rgba(0,0,0,.95),
             0 0 26px -8px rgba(0,229,199,.45),
             inset 0 1px 0 rgba(255,255,255,.07);
}
[data-testid="stMetricLabel"]{
  font-family:var(--dc-mono); font-size:.66rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--dc-muted); opacity:1;
}
[data-testid="stMetricValue"]{
  font-family:var(--dc-mono) !important;
  font-weight:600; color:#fff;
  text-shadow:0 0 24px rgba(0,229,199,.35);
}
[data-testid="stMetricDelta"]{font-family:var(--dc-mono);}

.card{
  position:relative;
  font-family:var(--dc-mono);
  background:linear-gradient(158deg, rgba(15,22,36,.66), rgba(11,17,29,.46));
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(0,229,199,.25);
  border-radius:14px;
  padding:16px 20px;
  margin-bottom:10px;
  line-height:1.65;
  color:var(--dc-text);
  box-shadow:0 18px 44px -30px rgba(0,0,0,.95),
             inset 0 1px 0 rgba(255,255,255,.05),
             0 0 24px -16px rgba(0,229,199,.8);
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.card:hover{
  transform:translateY(-1px);
  border-color:rgba(0,229,199,.45);
  box-shadow:0 22px 50px -28px rgba(0,0,0,.95),
             0 0 26px -10px rgba(0,229,199,.5);
}
.card b.k{
  font-family:var(--dc-mono); color:var(--dc-teal);
  font-size:.74rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase;
}
.card .coord, .coord, .conf{
  font-family:var(--dc-mono); color:#fff; letter-spacing:.02em;
}

.sevpill{
  display:inline-block;
  padding:3px 11px 4px;
  border-radius:999px;
  font-family:var(--dc-mono);
  font-size:.66rem;
  font-weight:700;
  line-height:1.25;
  letter-spacing:.14em;
  color:#06131b;
  border:1px solid rgba(6,19,27,.3);
  box-shadow:0 0 0 1px rgba(255,255,255,.18),
             inset 0 1px 0 rgba(255,255,255,.45),
             0 8px 18px -12px rgba(0,0,0,.95);
  transition:transform .18s ease, box-shadow .18s ease;
  vertical-align:middle;
}
.sevpill:hover{
  transform:translateY(-1px);
  box-shadow:0 0 0 1px rgba(255,255,255,.28),
             inset 0 1px 0 rgba(255,255,255,.45),
             0 0 18px -6px rgba(255,255,255,.6);
}

@keyframes dc-live-glow{
  0%,100%{
    box-shadow:0 0 0 1px rgba(0,229,199,.22), 0 0 20px -10px rgba(0,229,199,.55);
  }
  50%{
    box-shadow:0 0 0 1px rgba(0,229,199,.6), 0 0 36px -6px rgba(0,229,199,.9);
  }
}
.card.active-alert{
  border-color:rgba(0,229,199,.5);
  animation:dc-live-glow 2.4s ease-in-out infinite;
}
.card.active-alert::before{
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(0,229,199,.85), transparent);
}
@media (prefers-reduced-motion:reduce){
  .card.active-alert{animation:none;}
  [data-testid="stMetric"]:hover, .card:hover, .sevpill:hover{transform:none;}
}

[data-testid="stProgress"]{padding-bottom:.35rem;}
[data-testid="stProgressBarTrack"]{
  height:8px !important;
  border-radius:999px !important;
  background:rgba(130,168,200,.14) !important;
  box-shadow:inset 0 0 0 1px rgba(0,229,199,.18);
  overflow:hidden;
}
[data-testid="stProgressBarTrack"] > div{
  border-radius:999px;
  background:linear-gradient(90deg, rgba(0,229,199,.45), #00E5C7) !important;
  box-shadow:0 0 14px rgba(0,229,199,.6);
}

[data-testid="stDataFrame"]{
  border-radius:14px !important;
  overflow:hidden;
  border:1px solid rgba(0,229,199,.18);
  background:linear-gradient(158deg, rgba(15,22,36,.55), rgba(11,17,29,.4)) !important;
  box-shadow:0 18px 44px -32px rgba(0,0,0,.95);
}
[data-testid="stDataFrame"] [role="columnheader"]{
  background:rgba(0,229,199,.07) !important;
  border-bottom:1px solid rgba(0,229,199,.2) !important;
}
[data-testid="stDataFrame"] [role="gridcell"]{
  font-family:var(--dc-mono) !important;
  font-size:.76rem !important;
  background:transparent !important;
}
div[data-testid="stPlotlyChart"]{border-radius:14px; overflow:hidden;}

[data-testid="stAlert"]{
  background:rgba(15,22,36,.62) !important;
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  border-radius:12px !important;
  border-color:rgba(130,168,200,.25) !important;
  color:var(--dc-text) !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{color:var(--dc-text);}

.stButton > button{
  border-radius:10px !important;
  font-family:var(--dc-mono);
  font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; font-weight:600 !important;
  border:1px solid rgba(130,168,200,.25) !important;
  background:rgba(14,21,35,.75) !important;
  color:var(--dc-text) !important;
  transition:border-color .18s ease, box-shadow .2s ease,
             background .18s ease, color .18s ease;
}
.stButton > button:hover{
  border-color:rgba(0,229,199,.55) !important;
  color:#fff !important;
  background:rgba(0,229,199,.10) !important;
  box-shadow:0 0 22px -8px rgba(0,229,199,.75) !important;
}
.stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg, rgba(0,229,199,.92), rgba(14,158,140,.9)) !important;
  border-color:rgba(0,229,199,.8) !important;
  color:#04211D !important;
  box-shadow:0 8px 26px -12px rgba(0,229,199,.8) !important;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover{
  background:linear-gradient(135deg, #12F2D6, #0FB6A2) !important;
  box-shadow:0 12px 34px -8px rgba(0,229,199,.85) !important;
}

[data-baseweb="input"], [data-baseweb="select"] > div{
  background:rgba(10,15,25,.75) !important;
  border:1px solid rgba(130,168,200,.2) !important;
  border-radius:10px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within{
  border-color:rgba(0,229,199,.6) !important;
  box-shadow:0 0 0 1px rgba(0,229,199,.25), 0 0 20px -10px rgba(0,229,199,.8) !important;
}
[data-baseweb="input"] input, [data-baseweb="select"] input{
  font-family:var(--dc-mono) !important;
  color:var(--dc-text) !important;
  caret-color:var(--dc-teal) !important;
}
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label,
[data-testid="stSlider"] label, [data-testid="stTextInput"] label{
  font-family:var(--dc-mono) !important;
  font-size:.66rem !important;
  letter-spacing:.16em !important;
  text-transform:uppercase;
  color:var(--dc-faint) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:var(--dc-teal) !important;
  border:2px solid #060a12 !important;
  box-shadow:0 0 0 1px rgba(0,229,199,.6), 0 0 16px -2px rgba(0,229,199,.7) !important;
}
</style>
""", unsafe_allow_html=True)

st.title(":cyclone: Amplicast — Spatio-Temporal Weather Intelligence")
st.caption("GNN detection · diffusion downscaling · temporal tracker · "
           "MetPy physics · ERA5 reanalysis (May 2020, Bay of Bengal)")


def _api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, API_BASE_URL + path,
                                timeout=API_TIMEOUT, **kwargs)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"API unreachable at `{API_BASE_URL}` — {exc}. Start it with "
                 "`python -m scripts.serve_api`.")
        st.stop()
    return resp.json()


def _get(path: str, **params):
    return _api("GET", path,
                params={k: v for k, v in params.items() if v is not None})


def _post(path: str, payload: dict):
    return _api("POST", path, json=payload)


def _as_obj(d):
    return SimpleNamespace(**{
        k: _as_obj(v) if isinstance(v, dict) else v for k, v in d.items()})


def _records(items):
    return [_as_obj(r) for r in items]


def pill(sev: str) -> str:
    color = SEV_COLOR.get(sev, "#888")
    return f'<span class="sevpill" style="background:{color}">{sev.upper()}</span>'


def to_df(records) -> pd.DataFrame:
    rows = [{
        "id": r.id[:32], "type": r.type, "time": r.timestamp,
        "lat": r.lat, "lon": r.lon, "radius_km": round(r.radius_km, 1),
        "severity": r.severity, "confidence": round(r.confidence, 3),
        "lead_hours": r.lead_hours,
    } for r in records]
    return pd.DataFrame(rows)


def sev_marker(df):
    return [SEV_COLOR.get(s, "#888") for s in df["severity"]]


def load_track() -> list:
    return [{"kind": b["type"], "timestep_index": b["timestep_index"],
             "centre": {"lat": b["lat"], "lon": b["lon"]},
             "lead_hours": b["lead_hours"]}
            for b in _get("/api/forecast")]


def track_figure(track_raw, title):
    obs = [b for b in track_raw if b["kind"] == "observed"]
    fcs = [b for b in track_raw if b["kind"] == "forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=[b["centre"]["lat"] for b in obs],
        lon=[b["centre"]["lon"] for b in obs],
        mode="lines+markers", name="observed",
        line=dict(color="#00E5C7", width=3),
        marker=dict(size=9, color="#00E5C7", symbol="circle"),
        hovertemplate="obs %{lon:.2f}, %{lat:.2f}<extra></extra>"))
    fig.add_trace(go.Scattergeo(
        lat=[b["centre"]["lat"] for b in fcs],
        lon=[b["centre"]["lon"] for b in fcs],
        mode="lines+markers", name="forecast",
        line=dict(color="#FFB020", width=3, dash="dot"),
        marker=dict(size=8, color="#FF4D4D", symbol="x"),
        hovertemplate="fc %{lon:.2f}, %{lat:.2f}<extra></extra>"))
    if obs:
        fig.add_trace(go.Scattergeo(
            lat=[obs[0]["centre"]["lat"]], lon=[obs[0]["centre"]["lon"]],
            mode="markers+text", name="start",
            marker=dict(size=14, color="#00E5C7", symbol="star"),
            text=["start"], textposition="top center"))
    fig.update_layout(
        title=title,
        geo=dict(scope="asia", showland=True, landcolor="#0F1624",
                 coastlinecolor="#2c5f73", showcountries=False,
                 lakecolor="#070B12", oceancolor="#070B12",
                 projection_type="natural earth", lonaxis_range=[75, 100],
                 lataxis_range=[5, 30], bgcolor="#0A0E17"),
        paper_bgcolor="#0A0E17", font=dict(color="#dbe7f5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        height=560, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def alert_map_figure(df, title):
    fig = go.Figure()
    for sev in ("severe", "moderate", "low"):
        sub = df[df["severity"] == sev]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], name=sev,
            mode="markers",
            marker=dict(size=11, color=SEV_COLOR[sev], symbol="diamond"),
            customdata=sub["id"],
            hovertemplate="<b>%{customdata}</b><br>"
                          "(%{lon:.3f}, %{lat:.3f})<extra>"
                          + sev + "</extra>"))
    fig.update_layout(
        title=title,
        geo=dict(scope="asia", showland=True, landcolor="#0F1624",
                 coastlinecolor="#2c5f73", lakecolor="#070B12",
                 oceancolor="#070B12", projection_type="natural earth",
                 lonaxis_range=[75, 100], lataxis_range=[5, 30],
                 bgcolor="#0A0E17"),
        paper_bgcolor="#0A0E17", font=dict(color="#dbe7f5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        height=520, margin=dict(l=10, r=10, t=60, b=10))
    return fig


st.sidebar.markdown('<div class="sidebar-brand">AMPLICAST</div>'
                    '<div class="sidebar-brandsub">Weather Operations</div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("View", ["Overview", "Alerts", "Review Queue",
                                 "Forecast", "Exposure", "Ground Truth Track"])

if page == "Overview":
    alerts = _records(_get("/api/alerts", status="confirmed"))
    pend = _records(_get("/api/alerts/pending"))
    act = _records(_get("/api/alerts/active"))
    fc = _records(_get("/api/forecast"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed alerts", len(alerts))
    c2.metric("Pending review", len(pend))
    c3.metric("Forecast boxes", len(fc))
    c4.metric("Max lead",
              f"{max((f.lead_hours for f in fc), default=0)} h")

    l1, l2 = st.columns([3, 2])
    with l1:
        if act:
            a = act[0]
            st.markdown(
                f'<div class="card active-alert"><b class="k">ACTIVE ALERT</b> '
                f'{pill(a.severity)} &nbsp;·&nbsp; '
                f"confidence <b>{a.confidence:.3f}</b><br>"
                f"centre ({a.lat:.3f}, {a.lon:.3f}) · radius "
                f"{a.radius_km:.0f} km<br>"
                f"downscaled peak "
                f"{a.downscale.downscaled_peak:,.0f} (gain "
                f"{a.downscale.peak_gain:.2f})</div>",
                unsafe_allow_html=True)
            st.plotly_chart(alert_map_figure(to_df(alerts),
                                             "All detections "
                                             "on the Bay of Bengal"),
                            width="stretch")
        else:
            st.info("No active alerts yet.")
    with l2:
        st.markdown("#### Severity mix")
        if len(alerts):
            df = to_df(alerts)
            mix = df["severity"].value_counts().reindex(
                ["severe", "moderate", "low"]).fillna(0).astype(int)
            fig = go.Figure(go.Pie(
                labels=mix.index, values=mix.values,
                marker=dict(colors=[SEV_COLOR[s] for s in mix.index]),
                hole=0.62,
                textinfo="label+value"))
            fig.update_layout(paper_bgcolor="#0d1b2a",
                              font=dict(color="#dbe7f5"), height=300,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.markdown("#### Confidence timeline")
            fig = px.bar(df.sort_values("time"), x="time", y="confidence",
                         color="severity",
                         color_discrete_map=SEV_COLOR, height=260)
            fig.update_layout(paper_bgcolor="#0d1b2a", xaxis_tickangle=-60,
                              font=dict(color="#dbe7f5"),
                              legend=dict(orientation="h", y=1.15),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

elif page == "Alerts":
    a = st.sidebar.selectbox("Filter severity",
                             ["all", "low", "moderate", "severe"])
    s = st.sidebar.selectbox("Filter status",
                             ["confirmed", "pending_review", "rejected"],
                             index=0)
    recs = _records(_get("/api/alerts", status=s,
                         min_severity=None if a == "all" else a))
    df = to_df(recs)
    st.markdown(f"**{len(recs)}** alerts"
                + (f" · filter: {a}" if a != "all" else ""))
    st.dataframe(df.drop(columns=["type"]), width="stretch",
                 hide_index=True)
    sel = st.selectbox("Inspect alert", [r.id for r in recs])
    if sel:
        d = _as_obj(_get(f"/api/alerts/{sel}"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Severity", d.severity.upper())
        c2.metric("Confidence", f"{d.confidence:.3f}")
        c3.metric("Radius", f"{d.radius_km:.0f} km")
        st.markdown(
            f"<div class='card'>Centre "
            f"({d.lat:.4f}, {d.lon:.4f}) · timestep "
            f"{d.timestep_index} · {d.timestamp} · downscaled radius "
            f"{d.downscaled_radius_km} km</div>", unsafe_allow_html=True)
        cb = d.confidence_breakdown
        if cb:
            st.write("Confidence breakdown "
                     "(detection = 0.5·GNN + 0.5·ensemble agreement; "
                     "severity prob is a separate danger signal)")
            st.progress(min(cb.gnn_max_score, 1.0),
                        text=f"GNN max score {cb.gnn_max_score:.3f}")
            st.progress(min(cb.severity_prob_mean, 1.0),
                        text=f"severity prob {cb.severity_prob_mean:.3f}")
            st.write(f"ensemble {cb.n_ensemble} members · "
                     f"GNN mean {cb.gnn_mean_score:.3f}")
        if d.downscale:
            st.write("Downscale: coarse peak "
                     f"{d.downscale.coarse_peak:,.0f} → downscaled "
                     f"{d.downscale.downscaled_peak:,.0f} "
                     f"(peak gain {d.downscale.peak_gain:.3f})")

elif page == "Review Queue":
    st.markdown("#### Pending review — alerts must be confirmed before "
                "they are visible to the active-alert API (`/api/alerts/"
                "active`). Automated cycles land here by default.")
    pend = _records(_get("/api/alerts/pending"))
    if not pend:
        st.success("No alerts awaiting review.")
    else:
        st.warning(f"**{len(pend)}** alert(s) awaiting review.")
        ids = [r.id for r in pend]
        sel = st.selectbox("Alert", ids)
        d = _as_obj(_get(f"/api/alerts/{sel}"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", d.severity.upper())
        c2.metric("Confidence", f"{d.confidence:.3f}")
        c3.metric("Radius", f"{d.radius_km:.0f} km")
        c4.metric("Step", d.timestep_index)
        st.caption("Confidence = detection confidence "
                   "(0.5·GNN cluster score + 0.5·ensemble peak "
                   "agreement), not an inverse danger rating.")
        st.markdown(
            f"<div class='card'><b class='k'>{d.id}</b><br>centre "
            f"({d.lat:.4f}, {d.lon:.4f}) · {d.timestamp} · peak-gain "
            f"{d.downscale.peak_gain if d.downscale else 0:.3f}</div>",
            unsafe_allow_html=True)
        if d.confidence_breakdown:
            cb = d.confidence_breakdown
            st.progress(min(cb.gnn_max_score, 1.0),
                        text=f"GNN max score {cb.gnn_max_score:.3f}")
            st.progress(min(cb.severity_prob_mean, 1.0),
                        text=f"severity prob {cb.severity_prob_mean:.3f}")
        note = st.text_input("Reviewer note", key=f"note_{sel}")
        b1, b2 = st.columns(2)
        if b1.button("Confirm", type="primary"):
            _post(f"/api/alerts/{sel}/review",
                  {"decision": "confirmed", "note": note})
            st.rerun()
        if b2.button("Reject"):
            _post(f"/api/alerts/{sel}/review",
                  {"decision": "rejected", "note": note})
            st.rerun()

elif page == "Forecast":
    fc = _records(_get("/api/forecast"))
    df = to_df(fc).drop(columns=["type", "confidence"])
    st.dataframe(df, width="stretch", hide_index=True)
    st.plotly_chart(track_figure(load_track(),
                                 "Temporal-transformer forecast vs truth"),
                    width="stretch")

elif page == "Exposure":
    st.markdown("#### Point-risk assessment")
    lat = st.number_input("Latitude", value=13.5, min_value=-90.0,
                          max_value=90.0, step=0.1, format="%.2f")
    lon = st.number_input("Longitude", value=90.8, min_value=-180.0,
                          max_value=180.0, step=0.1, format="%.2f")
    radius = st.slider("Lookout radius (km)", 25, 500, 150)
    if st.button("Assess risk", type="primary"):
        res = _post("/api/exposure", {"lat": lat, "lon": lon,
                                      "radius_km": radius})
        col = {"none": "#5A6B80", "low": "#00E5C7", "medium": "#FFB020",
               "high": "#FF4D4D"}[res["risk"]]
        st.markdown(
            f"<h2 style='color:{col};margin:0;font-family:var(--dc-mono)'>"
            f"Risk: "
            f"{res['risk'].upper()} "
            f"<span style='font-size:1rem'>· score "
            f"{res['risk_score']:.3f}</span></h2>", unsafe_allow_html=True)
        if res["nearest"]:
            st.write("Nearest active zone:", res["nearest"])
        if res["contributing_alerts"]:
            st.dataframe(pd.DataFrame(res["contributing_alerts"]),
                         width="stretch", hide_index=True)
        else:
            st.info("No active alerts within reach of this point.")

else:
    st.plotly_chart(track_figure(load_track(),
                                 "Amplicast — Ground Truth Track "
                                 "(observed vs forecast)"),
                    width="stretch")
    raw = load_track()
    df = pd.DataFrame([{"lat": b["centre"]["lat"],
                        "lon": b["centre"]["lon"],
                        "kind": b["kind"]} for b in raw])
    st.dataframe(df, width="stretch", hide_index=True)

st.divider()
st.caption("Serves the Phase-7 API store (`outputs/phase4_alerts.json`, "
           "`outputs/phase5_amphan.json`, newest `outputs/live_run_*/`). "
           "Automated live alerts land in the Review Queue and only reach "
           "/api/alerts/active after a forecaster confirms them.")

st.sidebar.markdown('<div class="sidebar-live"><span class="dot"></span>'
                    'Live feed</div>', unsafe_allow_html=True)
