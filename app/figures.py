"""Plotly figure builders for Amplicast.

Every figure is themed from one template so the whole dashboard reads as a
single system: transparent plot surfaces (so the glass card behind shows
through), hairline cyan grids, Space Grotesk labels, and the shared
cyan/amber/red severity ramp.  Data access is unchanged — these functions
take the same frames/lists the API store produced.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from theme import (AMBER, FONT_MONO, FONT_SANS, MUTED, RED, SEV_COLOR, SLATE,
                   TEAL, TEXT)

SEV_ORDER = ["severe", "moderate", "low"]

# Shared dark plot template. Explicit colour keys only — Streamlit's plotly
# theme is disabled at the call site (theme=None) so nothing overrides these.
PLOT_THEME = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_SANS, size=11, color=MUTED),
    title=dict(font=dict(family=FONT_MONO, size=10, color=MUTED, textcase="upper")),
    colorway=[TEAL, AMBER, RED, SLATE],
    xaxis=dict(
        gridcolor="rgba(0,229,199,.09)", griddash="dot", zeroline=False,
        linecolor="rgba(0,229,199,.16)", tickfont=dict(family=FONT_MONO,
                                                       size=9, color=SLATE),
        title=dict(font=dict(family=FONT_MONO, size=9, color=SLATE, textcase="upper")),
        automargin=True),
    yaxis=dict(
        gridcolor="rgba(0,229,199,.09)", griddash="dot", zeroline=False,
        linecolor="rgba(0,229,199,.16)", tickfont=dict(family=FONT_MONO,
                                                       size=9, color=SLATE),
        title=dict(font=dict(family=FONT_MONO, size=9, color=SLATE, textcase="upper")),
        automargin=True),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
        font=dict(family=FONT_MONO, size=9, color=MUTED),
        bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant"),
    hoverlabel=dict(
        bgcolor="rgba(9,14,24,.96)", bordercolor="rgba(0,229,199,.42)",
        font=dict(family=FONT_MONO, size=10, color=TEXT),
        align="left"),
    margin=dict(l=8, r=8, t=8, b=8),
    separators=".,",
))

# Bay-of-Bengal operating area, unchanged from the original dashboard.
# The graticule lives on the lat/lon axis objects (Plotly 7 removed the
# geo-level showgraticule flag) — this is what gives the map its ops-chart
# measurement grid.
# Plotly 7's geo axis objects only accept showgrid/gridcolor/griddash/
# gridwidth/dtick/tick0 — no tickfont or axis line control.
_GRID = dict(showgrid=True, gridcolor="rgba(0,229,199,.075)",
             griddash="dot", gridwidth=0.6, dtick=5)

GEO_BASE = dict(
    scope="asia", projection_type="natural earth",
    lonaxis_range=[75, 100], lataxis_range=[5, 30],
    lonaxis=_GRID, lataxis=_GRID,
    showland=True, landcolor="#0E1826",
    showcoastlines=True, coastlinecolor="rgba(0,229,199,.32)", coastlinewidth=1,
    showcountries=True, countrycolor="rgba(0,229,199,.16)",
    showocean=True, oceancolor="#070B12",
    lakecolor="#070B12", showlakes=True,
    bgcolor="rgba(0,0,0,0)", showframe=False,
)

PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False,
               "scrollZoom": False, "doubleClick": False}


def _finish(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(template=PLOT_THEME, height=height,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6, r=6, t=10, b=6),
                      dragmode=False)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _mono(size: int = 8, color: str = SLATE):
    return dict(family=FONT_MONO, size=size, color=color)


# --------------------------------------------------------------------------
# Detection map
# --------------------------------------------------------------------------
def alert_map_figure(df, active_id: str | None = None, track=None,
                     height: int = 560):
    """All detections as glowing storm-core markers, colour-coded by severity.

    Each core is drawn as three layers so the marker reads as a lit beacon
    rather than a flat dot: a wide soft halo, a tight ring, and a crisp
    diamond core. ``active_id`` gets an extra outer ring for the beacon look.
    ``track`` (the best-track file) is overlaid as observed/predicted path so
    the map always shows where the core is going, not only where it is.
    """
    fig = go.Figure()

    if track:
        _draw_track(fig, track, context=True)

    for sev in SEV_ORDER:
        sub = df[df["severity"] == sev]
        if sub.empty:
            continue
        c = SEV_COLOR[sev]
        lat, lon, ids = sub["lat"], sub["lon"], sub["id"]
        # outer halo
        fig.add_trace(go.Scattergeo(
            lat=lat, lon=lon, mode="markers", name=f"{sev} · halo",
            showlegend=False, hoverinfo="skip",
            marker=dict(size=30, color=_rgba(c, 0.10), symbol="circle",
                        line=dict(width=0))))
        # ring
        fig.add_trace(go.Scattergeo(
            lat=lat, lon=lon, mode="markers", name=sev,
            showlegend=False, hoverinfo="skip",
            marker=dict(size=13, color=_rgba(c, 0.22), symbol="circle",
                        line=dict(width=1.1, color=_rgba(c, 0.75)))))
        # core
        fig.add_trace(go.Scattergeo(
            lat=lat, lon=lon, mode="markers", name=sev,
            legendgroup=sev, showlegend=True,
            marker=dict(size=7.5, color=c, symbol="diamond",
                        line=dict(width=1, color="rgba(4,10,16,.85)"),
                        opacity=0.98),
            customdata=list(ids),
            hovertemplate="<b>%{customdata}</b><br>"
                          "(%{lon:.3f}°E, %{lat:.3f}°N)"
                          "<extra>" + sev.upper() + "</extra>"))

    if active_id:
        hit = df[df["id"] == active_id]
        for _, row in hit.iterrows():
            c = SEV_COLOR.get(row["severity"], TEAL)
            for size, alpha, width in ((52, 0.05, 0), (30, 0.09, 0),
                                       (20, 0.0, 1.3)):
                fig.add_trace(go.Scattergeo(
                    lat=[row["lat"]], lon=[row["lon"]], mode="markers",
                    name="ACTIVE CORE", showlegend=False, hoverinfo="skip",
                    marker=dict(size=size, color=_rgba(c, alpha),
                                symbol="circle",
                                line=dict(width=width, color=_rgba(c, 0.65)))))

    fig.update_layout(geo=GEO_BASE)
    return _finish(fig, height)


# --------------------------------------------------------------------------
# Track: observed core path vs predicted path
# --------------------------------------------------------------------------
def _draw_track(fig: go.Figure, track_raw, context: bool = False) -> None:
    """Layer the best-track onto ``fig``.

    ``context=True`` is the map-overlay mode used by the detection lattice:
    path geometry only, no genesis star and no error annotation, so the
    detection markers stay the loudest thing on the chart.
    """
    obs = [b for b in track_raw if b["kind"] == "observed"]
    fcs = [b for b in track_raw if b["kind"] == "forecast"]
    olat = [b["centre"]["lat"] for b in obs]
    olon = [b["centre"]["lon"] for b in obs]
    flat = [b["centre"]["lat"] for b in fcs]
    flon = [b["centre"]["lon"] for b in fcs]
    flead = [b.get("lead_hours", 0) for b in fcs]
    ferr = [b.get("error_km") for b in fcs]
    legend = not context

    # glow underlays, then the crisp core lines
    if obs:
        fig.add_trace(go.Scattergeo(
            lat=olat, lon=olon, mode="lines", name="observed · glow",
            showlegend=False, hoverinfo="skip",
            line=dict(color=_rgba(TEAL, 0.16), width=10)))
    if fcs:
        fig.add_trace(go.Scattergeo(
            lat=flat, lon=flon, mode="lines", name="predicted · glow",
            showlegend=False, hoverinfo="skip",
            line=dict(color=_rgba(AMBER, 0.14), width=9, dash="dot")))

    # bridging connector: last observed -> first forecast
    if obs and fcs:
        fig.add_trace(go.Scattergeo(
            lat=[olat[-1], flat[0]], lon=[olon[-1], flon[0]], mode="lines",
            name="handover", showlegend=False, hoverinfo="skip",
            line=dict(color=_rgba(SLATE, 0.75), width=1.4, dash="dot")))

    # predicted cores: halo + hollow node
    if fcs:
        fig.add_trace(go.Scattergeo(
            lat=flat, lon=flon, mode="markers", name="predicted",
            legendgroup="predicted", showlegend=legend, hoverinfo="skip",
            marker=dict(size=15, color=_rgba(AMBER, 0.12), symbol="circle",
                        line=dict(width=0))))
        fig.add_trace(go.Scattergeo(
            lat=flat, lon=flon, mode="markers", name="predicted",
            legendgroup="predicted", showlegend=False,
            marker=dict(size=6, color="rgba(10,16,26,.9)", symbol="circle",
                        line=dict(width=1.5, color=AMBER)),
            customdata=[f"+{h}h" for h in flead],
            hovertemplate="<b>T+%{customdata}</b><br>"
                          "(%{lon:.3f}°E, %{lat:.3f}°N)"
                          "<extra>PREDICTED</extra>"))

    # observed cores
    if obs:
        fig.add_trace(go.Scattergeo(
            lat=olat, lon=olon, mode="markers", name="observed",
            legendgroup="observed", showlegend=legend, hoverinfo="skip",
            marker=dict(size=17, color=_rgba(TEAL, 0.15), symbol="circle",
                        line=dict(width=0))))
        fig.add_trace(go.Scattergeo(
            lat=olat, lon=olon, mode="markers", name="observed",
            legendgroup="observed", showlegend=False,
            marker=dict(size=7, color=TEAL, symbol="diamond",
                        line=dict(width=1, color="rgba(4,10,16,.9)")),
            customdata=[str(b.get("time", ""))[:16] for b in obs],
            hovertemplate="<b>OBSERVED</b><br>%{customdata}<br>"
                          "(%{lon:.3f}°E, %{lat:.3f}°N)<extra></extra>"))

    if context:
        return

    # genesis marker
    if obs:
        fig.add_trace(go.Scattergeo(
            lat=[olat[0]], lon=[olon[0]], mode="markers", name="genesis",
            showlegend=False, hoverinfo="skip",
            marker=dict(size=30, color=_rgba(TEAL, 0.13), symbol="circle",
                        line=dict(width=0))))
        fig.add_trace(go.Scattergeo(
            lat=[olat[0]], lon=[olon[0]], mode="markers+text",
            name="genesis", showlegend=False,
            marker=dict(size=11, color=TEAL, symbol="star",
                        line=dict(width=1, color="rgba(4,10,16,.9)")),
            text=["GENESIS"], textposition="top center",
            textfont=_mono(8, TEAL)))

    # forecast-error annotation on the worst forecast step
    worst = [i for i, e in enumerate(ferr) if e]
    if worst:
        i = worst[-1]
        fig.add_trace(go.Scattergeo(
            lat=[flat[i]], lon=[flon[i]], mode="markers+text",
            showlegend=False, hoverinfo="skip",
            marker=dict(size=9, color=RED, symbol="x-thin",
                        line=dict(width=1.8, color=RED)),
            text=[f"{ferr[i]:.0f} km err"], textposition="middle right",
            textfont=_mono(8, RED)))


def track_figure(track_raw, height: int = 580):
    """Best-track: solid observed cores, dashed predicted path, origin star."""
    fig = go.Figure()
    _draw_track(fig, track_raw)
    fig.update_layout(geo=GEO_BASE)
    return _finish(fig, height)


# --------------------------------------------------------------------------
# Severity mix donut
# --------------------------------------------------------------------------
def severity_donut(mix, total: int, height: int = 268):
    """Severity composition. Same ramp as every other chart in the app."""
    labels = [str(s).upper() for s in mix.index]
    values = [int(v) for v in mix.values]
    colors = [SEV_COLOR.get(s, SLATE) for s in labels]
    present = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not present:
        present = [("no data", 1, SLATE)]

    fig = go.Figure(go.Pie(
        labels=[p[0] for p in present], values=[p[1] for p in present],
        hole=0.70, sort=False, direction="clockwise",
        marker=dict(colors=[p[2] for p in present],
                    line=dict(color="rgba(9,14,24,.95)", width=2)),
        textinfo="none", hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>%{value} alerts"
                      "<br>%{percent}<extra></extra>"))
    # faint instrument ring behind the donut, so the hole is not empty space
    fig.add_trace(go.Pie(
        labels=["ring"], values=[1], hole=0.86, sort=False,
        marker=dict(colors=["rgba(0,229,199,.07)"],
                    line=dict(color="rgba(0,229,199,.16)", width=1)),
        textinfo="none", hoverinfo="skip", showlegend=False, opacity=0.9))

    fig.update_layout(
        annotations=[dict(
            x=0.5, y=0.47, xref="paper", yref="paper", showarrow=False,
            text=f"<b style='font-family:{FONT_MONO};font-size:21px;"
                 f"color:{TEXT}'>{total}</b><br>"
                 f"<span style='font-family:{FONT_MONO};font-size:8px;"
                 f"letter-spacing:2px;color:{SLATE}'>ALERTS</span>",
            align="center")],
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02,
                    xanchor="center", x=0.5,
                    font=dict(family=FONT_MONO, size=9, color=MUTED)),
        margin=dict(l=4, r=4, t=4, b=30))
    return _finish(fig, height)


# --------------------------------------------------------------------------
# Confidence timeline
# --------------------------------------------------------------------------
def confidence_timeline(df, height: int = 262):
    """Detection confidence per valid time, bars coloured by severity."""
    ordered = df.sort_values("time")
    mean_conf = float(ordered["confidence"].mean()) if len(ordered) else 0.0
    fig = px.bar(ordered, x="time", y="confidence", color="severity",
                 color_discrete_map=SEV_COLOR,
                 category_orders={"severity": SEV_ORDER},
                 hover_data={"severity": True, "time": True,
                             "confidence": ":.3f"})
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=1.5),
        opacity=0.92,
        hovertemplate="<b>%{x}</b><br>confidence %{y:.3f}"
                      "<extra>%{fullData.name}</extra>")
    # cohort mean as a dashed reference — the bar/line comparison is the
    # quickest read on whether the window is degrading
    fig.add_hline(y=mean_conf, line=dict(color=_rgba(TEAL, 0.55), width=1,
                                         dash="dot"),
                  annotation_text=f"mean {mean_conf:.3f}",
                  annotation_position="top left",
                  annotation_font=_mono(8, TEAL))
    fig.update_layout(
        bargap=0.42,
        xaxis=dict(tickangle=-58, showgrid=False,
                   tickfont=dict(family=FONT_MONO, size=8)),
        yaxis=dict(range=[0, 1.0], dtick=0.25, gridcolor="rgba(0,229,199,.09)",
                   griddash="dot",
                   title=dict(text="CONFIDENCE", font=_mono(8, SLATE))),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    font=dict(family=FONT_MONO, size=8, color=MUTED)))
    return _finish(fig, height)


# --------------------------------------------------------------------------
# Track error drift (forecast view)
# --------------------------------------------------------------------------
def track_error_figure(fc, height: int = 250):
    """Forecast error in km against lead time — the model's honest scorecard.

    The error line is tinted along the severity ramp (teal while the track is
    inside tolerance, amber as it drifts, red once the forecast is being
    corrected by hand), so this chart speaks the same colour language as the
    alert register.
    """
    pts = [(f.lead_hours, f.track_error_km)
           for f in fc if f.track_error_km]
    if len(pts) < 2:
        return None
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    tol = max(ys) * 0.5

    def _ramp(v: float) -> str:
        if v <= tol:
            return TEAL
        return AMBER if v <= max(ys) * 0.8 else RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="track error",
        line=dict(color=_rgba(TEAL, 0.18), width=11),
        hoverinfo="skip", showlegend=False))
    # one segment per colour band so the ramp follows the data
    for i in range(len(xs) - 1):
        seg_y = [ys[i], ys[i + 1]]
        fig.add_trace(go.Scatter(
            x=[xs[i], xs[i + 1]], y=seg_y, mode="lines",
            showlegend=False, hoverinfo="skip",
            line=dict(color=_ramp(max(seg_y)), width=1.8,
                      shape="spline", smoothing=0.5)))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="track error",
        showlegend=False,
        marker=dict(size=6, color=[_ramp(v) for v in ys], symbol="circle",
                    line=dict(width=1, color="rgba(9,14,24,.9)")),
        hovertemplate="<b>+%{x}h lead</b><br>%{y:.1f} km error"
                      "<extra></extra>"))
    # tolerance ceiling
    fig.add_hline(y=ys[-1], line=dict(color=_rgba(RED, 0.45), width=1,
                                      dash="dot"),
                  annotation_text=f"peak {ys[-1]:.0f} km",
                  annotation_position="bottom right",
                  annotation_font=_mono(8, RED))
    fig.update_layout(
        xaxis=dict(title=dict(text="LEAD HOURS", font=_mono(8, SLATE)),
                   showgrid=False, zeroline=False),
        yaxis=dict(title=dict(text="ERROR KM", font=_mono(8, SLATE)),
                   gridcolor="rgba(0,229,199,.09)", griddash="dot",
                   rangemode="tozero"),
        showlegend=False, margin=dict(l=6, r=10, t=10, b=6))
    return _finish(fig, height)
