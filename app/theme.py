"""Amplicast visual system — "mission control" identity.

Design tokens, the injected stylesheet, and the small HTML component library
used by ``app/dashboard.py``.  Nothing in here touches the data layer: the
module only turns already-computed values into markup and CSS.

The identity in one paragraph: near-black space-navy shell with a non-flat
layered background (aurora washes + survey grid + faint bathymetric contours),
translucent glass panels with hairline cyan borders and soft glow instead of
drop shadows, Space Grotesk for prose and JetBrains Mono for every number a
human has to read under pressure.  Colour is a severity channel and nothing
else: cyan = nominal/confirmed, amber = watch/pending, red = act/critical.

Stack: Streamlit >= 1.64 (``st.html`` + ``container(key=...)`` hooks),
Plotly, and a Google Fonts import that degrades to the system stack offline.
"""

from __future__ import annotations

import math
from urllib.parse import quote

# --------------------------------------------------------------------------
# 1. Palette
# --------------------------------------------------------------------------
# Severity is the single most important colour signal in the product, so it
# maps onto the alarm convention ops rooms already read: cyan = nominal,
# amber = watch, red = act.  Exposure risk borrows the same ramp.
BG_DEEP = "#060911"
BG_0 = "#0A0E17"
BG_1 = "#0D1220"

TEAL = "#00E5C7"
TEAL_DIM = "#0E9E8C"
AMBER = "#FFB020"
RED = "#FF4D4D"
SLATE = "#5A6B80"
BLUE = "#4C7DFF"

TEXT = "#E6EDF6"
MUTED = "#8FA1B7"
FAINT = "#61748C"

SEV_COLOR = {"low": TEAL, "moderate": AMBER, "severe": RED,
             "unknown": SLATE}
SEV_RANK = {"low": 1, "moderate": 2, "severe": 3}

RISK_COLOR = {"none": SLATE, "low": TEAL, "medium": AMBER, "high": RED}

FONT_SANS = ("'Space Grotesk','Inter',system-ui,-apple-system,"
             "'Segoe UI',Roboto,sans-serif")
FONT_MONO = ("'JetBrains Mono','Space Mono',ui-monospace,'SF Mono',"
             "Consolas,'Liberation Mono',monospace")


# --------------------------------------------------------------------------
# 2. Inline stroke icon set (24x24, no external dependency)
# --------------------------------------------------------------------------
ICON_PATHS = {
    "overview": "<circle cx='12' cy='12' r='8.2'/><circle cx='12' cy='12' r='3.1'/>"
                "<path d='M12 12 18.2 5.8'/><path d='M12 3.2v1.8M12 19v1.8M3.2 12H5M19 12h1.8'/>",
    "alerts": "<path d='M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20.2h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z'/>"
              "<path d='M12 9.2v4.2'/><path d='M12 16.9h.01'/>",
    "review": "<path d='M9.2 4H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2.2'/>"
              "<rect x='9' y='2.4' width='6' height='3.3' rx='1'/>"
              "<path d='m9 13.4 2.1 2.1 4.4-4.4'/>",
    "forecast": "<path d='M3 18.4c6.2 0 5.4-12.8 12-12.8h3.2'/>"
                "<path d='M15.8 2.6 19.4 5.6 15.8 8.6'/>"
                "<circle cx='3' cy='18.4' r='1.7'/>",
    "exposure": "<circle cx='12' cy='12' r='8'/><circle cx='12' cy='12' r='3.1'/>"
                "<path d='M12 1.7v3M12 19.3v3M1.7 12h3M19.3 12h3'/>",
    "truth": "<ellipse cx='12' cy='6' rx='7.2' ry='3'/>"
             "<path d='M4.8 6v6c0 1.7 3.2 3 7.2 3s7.2-1.3 7.2-3V6'/>"
             "<path d='M4.8 12v6c0 1.7 3.2 3 7.2 3s7.2-1.3 7.2-3v-6'/>",
    "layers": "<path d='m12 2.9 8.6 4.6L12 12.1 3.4 7.5 12 2.9Z'/>"
              "<path d='m3.4 12.1 8.6 4.6 8.6-4.6'/>"
              "<path d='m3.4 16.7 8.6 4.6 8.6-4.6'/>",
    "waypoints": "<rect x='2.6' y='3.8' width='7.4' height='7.4' rx='1.4'/>"
                 "<rect x='14' y='12.8' width='7.4' height='7.4' rx='1.4'/>"
                 "<path d='M10 7.5h2.2a1.8 1.8 0 0 1 1.8 1.8v3.5'/>",
    "clock": "<circle cx='12' cy='12' r='8.4'/><path d='M12 7.2V12l3.1 1.9'/>",
    "check": "<path d='m4.8 12.6 4.7 4.7L19.4 7'/>",
    "close": "<path d='M6.2 6.2 17.8 17.8M17.8 6.2 6.2 17.8'/>",
    "shield": "<path d='M12 2.9 5 5.9v5.4c0 4.4 2.9 8.5 7 9.8 4.1-1.3 7-5.4 7-9.8V5.9l-7-3Z'/>"
              "<path d='m9.2 11.9 2 2 3.6-3.8'/>",
    "pin": "<path d='M12 21.2s7-5.7 7-11.2a7 7 0 1 0-14 0c0 5.5 7 11.2 7 11.2Z'/>"
           "<circle cx='12' cy='9.8' r='2.6'/>",
    "gauge": "<path d='M3.4 18.4a8.6 8.6 0 1 1 17.2 0'/><path d='m12 18.4 4.3-5.4'/>"
             "<path d='M12 18.4h.01'/>",
    "wave": "<path d='M2 12c2.4-4 4.2-4 6.6 0s4.2 4 6.6 0 4.2-4 6.6 0'/>",
    "signal": "<path d='M8.6 15.3a4.8 4.8 0 0 1 6.8 0'/>"
              "<path d='M5.6 12.3a8.8 8.8 0 0 1 12.8 0'/>"
              "<path d='M2.6 9.3a12.8 12.8 0 0 1 18.8 0'/>",
    "search": "<circle cx='11' cy='11' r='6.6'/><path d='m16 16 4.4 4.4'/>",
    "lock": "<rect x='4.6' y='10.4' width='14.8' height='10' rx='2.2'/>"
            "<path d='M8.2 10.4V7.6a3.8 3.8 0 0 1 7.6 0v2.8'/>",
    "flask": "<path d='M9.4 2.9v6.4L4.2 18.4a2 2 0 0 0 1.7 3.1h12.2a2 2 0 0 0 1.7-3.1l-5.2-9.1V2.9'/>"
             "<path d='M8.2 2.9h7.6'/><path d='M6.6 15.4h10.8'/>",
    "satellite": "<path d='m8.4 15.6 4-4'/><path d='M13.4 10.6 9 6.2'/>"
                 "<path d='M6.6 4.2 4.2 6.6l3.2 3.2 2.4-2.4z'/>"
                 "<path d='m17.4 6.6 2.4 2.4-8.6 8.6-2.4-2.4z'/>"
                 "<path d='M14.2 16.4 12.4 18.2M18 17.2l-1.4 1.4'/>",
    "radar": "<circle cx='12' cy='12' r='8.4'/><circle cx='12' cy='12' r='3.6'/>"
             "<path d='M12 12 18.6 5.4'/>",
    "grid": "<path d='M3.4 3.4h6.4v6.4H3.4zM14.2 3.4h6.4v6.4h-6.4zM3.4 14.2h6.4v6.4H3.4zM14.2 14.2h6.4v6.4h-6.4z'/>",
    "cpu": "<rect x='6.4' y='6.4' width='11.2' height='11.2' rx='1.6'/>"
           "<path d='M9.6 2.6v3.8M14.4 2.6v3.8M9.6 17.6v3.8M14.4 17.6v3.8M2.6 9.6h3.8M2.6 14.4h3.8M17.6 9.6h3.8M17.6 14.4h3.8'/>",
    "sat": "<circle cx='12' cy='12' r='3.2'/>"
           "<path d='M12 4.4a7.6 7.6 0 0 1 7.6 7.6M12 19.6A7.6 7.6 0 0 1 4.4 12'/>"
           "<path d='M12 1.4a10.6 10.6 0 0 1 10.6 10.6M12 22.6A10.6 10.6 0 0 1 1.4 12'/>",
    "arrow": "<path d='M4.4 12h14.2M13 6.4 18.6 12 13 17.6'/>",
}


def icon(name: str, size: int = 16, cls: str = "", stroke: str = "currentColor",
         width: float = 1.7) -> str:
    """Inline SVG for a named icon; inherits colour via ``currentColor``."""
    body = ICON_PATHS.get(name, ICON_PATHS["gauge"])
    return (f"<svg class='{cls}' width='{size}' height='{size}' viewBox='0 0 24 24' "
            f"fill='none' stroke='{stroke}' stroke-width='{width}' "
            f"stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
            f"{body}</svg>")


def _icon_url(name: str, color: str) -> str:
    """``url(...)`` token for use inside an injected stylesheet."""
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
           f"stroke='{color}' stroke-width='1.7' stroke-linecap='round' "
           f"stroke-linejoin='round'>{ICON_PATHS.get(name, ICON_PATHS['gauge'])}</svg>")
    return f'url("data:image/svg+xml;utf8,{quote(svg, safe="")}")'


NAV_ITEMS = [
    ("Overview", "overview"),
    ("Alerts", "alerts"),
    ("Review Queue", "review"),
    ("Forecast", "forecast"),
    ("Exposure", "exposure"),
    ("Ground Truth", "truth"),
]


# --------------------------------------------------------------------------
# 3. Texture assets
# --------------------------------------------------------------------------
def _contour_tile() -> str:
    """Seamless bathymetric-contour tile, used as a very low-opacity layer.

    Ops charts sit on bathymetry, so the shell carries the same vocabulary:
    nested isobath-like rings rather than a generic dotted texture.
    """
    paths = [
        "M-20 150 C 60 96 118 214 208 168 S 356 92 460 158",
        "M-20 250 C 74 196 122 316 226 262 S 372 186 460 258",
        "M-20 350 C 88 300 128 424 244 366 S 384 292 460 360",
        "M-20 62 C 44 20 108 128 190 86 S 340 18 460 92",
        "M160 -20 C 200 56 92 108 148 196 S 268 292 208 440",
        "M268 -20 C 310 62 198 118 254 204 S 376 300 316 440",
    ]
    body = "".join(f"<path d='{d}'/>" for d in paths)
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='480' height='420' "
           f"viewBox='0 0 480 420'><g fill='none' stroke='#7FE9DA' "
           f"stroke-width='1'>{body}</g></svg>")
    return f'url("data:image/svg+xml;utf8,{quote(svg, safe="")}")'


CONTOUR_LAYER = _contour_tile()


# --------------------------------------------------------------------------
# 4. Stylesheet
# --------------------------------------------------------------------------
def _nav_css() -> str:
    """Per-item icon tokens for the sidebar radio, injected as CSS."""
    out = []
    for i, (_, key) in enumerate(NAV_ITEMS, start=1):
        out.append(
            f".ac-nav [data-testid='stRadio'] "
            f"label:nth-of-type({i}){{--ico:{_icon_url(key, MUTED)};"
            f"--ico-on:{_icon_url(key, TEAL)}}}"
        )
    return "\n".join(out)


_CSS = """
/* ============ 0 · fonts ============ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

/* ============ 1 · tokens ============ */
:root{
  --ac-bg-deep:#060911; --ac-bg-0:#0A0E17; --ac-bg-1:#0D1220;
  --ac-teal:#00E5C7; --ac-teal-dim:#0E9E8C; --ac-amber:#FFB020;
  --ac-red:#FF4D4D; --ac-slate:#5A6B80; --ac-blue:#4C7DFF;
  --ac-text:#E6EDF6; --ac-muted:#8FA1B7; --ac-faint:#61748C;
  --ac-glass:rgba(15,22,36,.66);
  --ac-glass-2:rgba(11,17,29,.50);
  --ac-hair:rgba(0,229,199,.16);
  --ac-hair-soft:rgba(130,168,200,.12);
  --ac-glow:rgba(0,229,199,.45);
  --ac-sans:'Space Grotesk','Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --ac-mono:'JetBrains Mono','Space Mono',ui-monospace,'SF Mono',Consolas,'Liberation Mono',monospace;
  --ac-r:14px; --ac-r-sm:10px;
}

/* ============ 2 · shell: layered, never flat ============ */
html, body, [data-testid="stAppViewContainer"], .stApp,
[data-testid="stMain"], .stMain{ background:transparent !important; }

body{
  font-family:var(--ac-sans);
  color:var(--ac-text);
  background-color:var(--ac-bg-deep);
  background-image:
    radial-gradient(1400px 760px at 8% -14%, rgba(0,229,199,.13), transparent 60%),
    radial-gradient(1180px 700px at 96% -6%, rgba(76,125,255,.12), transparent 58%),
    radial-gradient(900px 620px at 52% 116%, rgba(255,176,32,.06), transparent 62%),
    radial-gradient(760px 520px at 88% 96%, rgba(0,229,199,.05), transparent 60%),
    repeating-linear-gradient(0deg, rgba(120,190,220,.032) 0 1px, transparent 1px 48px),
    repeating-linear-gradient(90deg, rgba(120,190,220,.032) 0 1px, transparent 1px 48px),
    linear-gradient(180deg,#0A0E17 0%,#070A11 44%,#090D16 100%);
  background-attachment:fixed, fixed, fixed, fixed, fixed, fixed, fixed;
}
/* survey grid: every 4th line is brighter, like a chart overlay */
body::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    repeating-linear-gradient(0deg, rgba(0,229,199,.055) 0 1px, transparent 1px 192px),
    repeating-linear-gradient(90deg, rgba(0,229,199,.055) 0 1px, transparent 1px 192px),
    """ + CONTOUR_LAYER + """;
  background-size:auto, auto, 480px 420px;
  opacity:.5; mix-blend-mode:screen;
}
/* vignette so glass panels read as floating over depth */
body::after{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:1;
  background:radial-gradient(122% 88% at 50% 40%, transparent 50%, rgba(0,0,0,.46) 100%);
}
[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none !important;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{background:transparent;}

[data-testid="stMainBlockContainer"],[data-testid="stBlockContainer"],
.block-container{
  max-width:100% !important; padding:.4rem 1.5rem 2.4rem !important;
  position:relative; z-index:2;
}
[data-testid="stVerticalBlock"]{gap:.7rem;}
[data-testid="stHorizontalBlock"]{gap:.9rem;}
[data-testid="stAppViewContainer"] > .main{background:transparent;}

/* page furniture ---------------------------------------------- */
#MainMenu, footer, [data-testid="stFooter"]{visibility:hidden;height:0;}
hr, [data-testid="stDivider"]{border-color:var(--ac-hair-soft) !important;opacity:.7;}

h1,h2,h3,h4,h5{font-family:var(--ac-sans);letter-spacing:-.01em;font-weight:600;}
p,span,li,label,summary{font-family:var(--ac-sans);}
a{color:var(--ac-teal);text-decoration:none;}
[data-testid="stCaptionContainer"], .stCaption, small,
[data-testid="stMarkdownContainer"] p{color:var(--ac-muted);}
code, kbd, pre{
  font-family:var(--ac-mono) !important;
  background:rgba(0,229,199,.07) !important; color:#9FF0E2 !important;
  border:1px solid var(--ac-hair); border-radius:5px; padding:.5px 5px;
}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-track{background:rgba(255,255,255,.02);}
::-webkit-scrollbar-thumb{
  background:linear-gradient(180deg,rgba(0,229,199,.30),rgba(0,229,199,.13));
  border-radius:9px; border:2px solid transparent; background-clip:padding-box;
}
::-webkit-scrollbar-thumb:hover{background:rgba(0,229,199,.5);background-clip:padding-box;}

/* ============ 3 · sidebar ============ */
[data-testid="stSidebar"],[data-testid="stSidebarContainer"]{
  background:linear-gradient(180deg, rgba(10,15,25,.92), rgba(6,9,16,.95)) !important;
  backdrop-filter:blur(20px) saturate(135%);
  -webkit-backdrop-filter:blur(20px) saturate(135%);
  border-right:1px solid var(--ac-hair);
  box-shadow:1px 0 0 rgba(0,0,0,.55), 30px 0 70px -46px var(--ac-glow);
  min-width:272px; max-width:272px;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:.5rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.5rem;}

.ac-brand{display:flex;align-items:center;gap:11px;padding:.35rem .15rem .7rem;}
.ac-mark{
  position:relative; overflow:hidden;
  width:38px;height:38px;flex:0 0 38px;border-radius:11px;
  display:grid;place-items:center;color:var(--ac-teal);
  background:linear-gradient(150deg, rgba(0,229,199,.20), rgba(0,229,199,.04));
  border:1px solid rgba(0,229,199,.34);
  box-shadow:0 0 18px -4px var(--ac-glow), inset 0 0 18px -8px var(--ac-glow);
}
/* slow conic sweep inside the brand mark = "system is scanning" */
.ac-mark::after{
  content:""; position:absolute; inset:-40%;
  background:conic-gradient(from 0deg, rgba(0,229,199,0) 0deg, rgba(0,229,199,0) 300deg,
              rgba(0,229,199,.28) 350deg, rgba(0,229,199,0) 360deg);
  animation:ac-spin 6.5s linear infinite;
}
.ac-mark svg{position:relative; z-index:1; filter:drop-shadow(0 0 5px var(--ac-glow));}
.ac-name{font-family:var(--ac-sans);font-weight:700;font-size:1.06rem;
  letter-spacing:.20em;line-height:1;color:#fff;}
.ac-tag{font-family:var(--ac-mono);font-size:.60rem;letter-spacing:.22em;
  color:var(--ac-teal);opacity:.85;margin-top:5px;}

.ac-nav-eyebrow,.ac-side-eyebrow{
  font-family:var(--ac-mono);font-size:.60rem;letter-spacing:.24em;
  color:var(--ac-faint);text-transform:uppercase;
  padding:.55rem .55rem .3rem;display:flex;align-items:center;gap:.5rem;
}
.ac-nav-eyebrow::after,.ac-side-eyebrow::after{
  content:"";flex:1;height:1px;
  background:linear-gradient(90deg, var(--ac-hair), transparent);
}

.ac-nav [data-testid="stRadio"] > div:last-child,
.ac-nav [data-testid="stRadio"] > div{padding:0 !important;}
.ac-nav [data-testid="stRadio"] [role="radiogroup"]{gap:.14rem !important;}
.ac-nav [data-testid="stRadio"] label{
  position:relative; width:100%; cursor:pointer;
  display:flex; align-items:center; gap:.7rem;
  padding:.56rem .6rem .56rem .78rem !important;
  border-radius:0 var(--ac-r-sm) var(--ac-r-sm) 0;
  color:var(--ac-muted); background:transparent;
  transition:background .18s ease, color .18s ease;
  min-height:40px;
}
.ac-nav [data-testid="stRadio"] label > div{width:100%;}
.ac-nav [data-testid="stRadio"] label p{margin:0;line-height:1.2;
  font-size:.855rem;font-weight:500;letter-spacing:.01em;}
.ac-nav [data-testid="stRadio"] label::before{
  content:var(--ico); width:17px;height:17px;flex:0 0 17px;
  filter:opacity(.75); transition:filter .2s ease, transform .2s ease;
}
/* glowing left-edge accent bar */
.ac-nav [data-testid="stRadio"] label::after{
  content:""; position:absolute; left:0; top:50%; transform:translateY(-50%);
  width:3px; height:0; border-radius:0 3px 3px 0; background:var(--ac-teal);
  box-shadow:0 0 10px 1px var(--ac-glow), 0 0 22px 2px rgba(0,229,199,.25);
  transition:height .24s cubic-bezier(.4,0,.2,1);
}
.ac-nav [data-testid="stRadio"] label:hover{color:var(--ac-text);
  background:linear-gradient(90deg, rgba(0,229,199,.06), transparent 75%);}
.ac-nav [data-testid="stRadio"] label:hover::before{filter:opacity(1);
  transform:scale(1.08);}
.ac-nav [data-testid="stRadio"] label:has(input:checked){
  color:#fff; font-weight:500;
  background:linear-gradient(90deg, rgba(0,229,199,.15), rgba(0,229,199,.015) 78%);
}
.ac-nav [data-testid="stRadio"] label:has(input:checked)::after{height:58%;}
.ac-nav [data-testid="stRadio"] label:has(input:checked)::before{
  content:var(--ico-on); filter:drop-shadow(0 0 5px var(--ac-glow));}
.ac-nav [data-testid="stRadio"] input{
  position:absolute; opacity:0; width:1px; height:1px; pointer-events:none;
}
.ac-nav [data-testid="stRadio"] label:has(input:focus-visible){
  outline:1px solid var(--ac-teal); outline-offset:-1px;
}

/* sidebar live block ------------------------------------------ */
.ac-live{
  display:flex;align-items:center;gap:.5rem;margin:.15rem .3rem .1rem;
  padding:.42rem .55rem;border-radius:var(--ac-r-sm);
  background:rgba(0,229,199,.05);border:1px solid var(--ac-hair);
}
.ac-dot{width:7px;height:7px;border-radius:50%;background:var(--ac-teal);flex:0 0 7px;
  position:relative;box-shadow:0 0 8px var(--ac-teal);}
.ac-dot::after{
  content:""; position:absolute; inset:-4px; border-radius:50%;
  border:1px solid currentColor; opacity:.55;
  animation:ac-pulse 2.2s ease-out infinite;
}
.ac-live.warn .ac-dot{background:var(--ac-amber);box-shadow:0 0 8px var(--ac-amber);}
.ac-live.crit .ac-dot{background:var(--ac-red);box-shadow:0 0 9px var(--ac-red);
  animation:ac-blink 1.15s ease-in-out infinite;}
.ac-live span{font-family:var(--ac-mono);font-size:.63rem;letter-spacing:.13em;
  color:var(--ac-text);opacity:.92;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

.ac-note{
  font-family:var(--ac-mono);font-size:.585rem;line-height:1.65;letter-spacing:.055em;
  color:var(--ac-faint);text-transform:uppercase;
  padding:.6rem .55rem;border-left:2px solid rgba(255,176,32,.5);
  background:linear-gradient(90deg, rgba(255,176,32,.07), transparent 85%);
  border-radius:0 var(--ac-r-sm) var(--ac-r-sm) 0;
}
/* sidebar system readout */
.ac-sys{display:flex;flex-direction:column;gap:.3rem;padding:.2rem .55rem .3rem;}
.ac-sys-row{display:flex;align-items:center;gap:.45rem;
  font-family:var(--ac-mono);font-size:.585rem;letter-spacing:.09em;}
.ac-sys-row i{width:5px;height:5px;border-radius:50%;background:var(--ac-teal);
  box-shadow:0 0 7px var(--ac-teal);flex:0 0 5px;}
.ac-sys-row b{color:var(--ac-muted);font-weight:400;text-transform:uppercase;}
.ac-sys-row span{margin-left:auto;color:var(--ac-faint);}

/* ============ 4 · glass panels ============ */
.ac-card{
  position:relative; border-radius:var(--ac-r);
  background:linear-gradient(158deg, var(--ac-glass), var(--ac-glass-2));
  backdrop-filter:blur(16px) saturate(125%);
  -webkit-backdrop-filter:blur(16px) saturate(125%);
  border:1px solid var(--ac-hair-soft);
  box-shadow:0 18px 44px -30px rgba(0,0,0,.95), inset 0 1px 0 rgba(255,255,255,.05);
  padding:.95rem 1.05rem; overflow:hidden;
  transition:border-color .22s ease, box-shadow .22s ease, transform .22s ease;
}
.ac-card::before{
  content:""; position:absolute; inset:0; pointer-events:none; opacity:0;
  background:radial-gradient(420px 130px at 12% -10%, rgba(0,229,199,.16), transparent 70%);
  transition:opacity .25s ease;
}
.ac-card:hover{
  border-color:rgba(0,229,199,.30);
  box-shadow:0 22px 50px -28px rgba(0,0,0,.95),
             0 0 0 1px rgba(0,229,199,.08),
             0 10px 44px -24px var(--ac-glow);
  transform:translateY(-1px);
}
.ac-card:hover::before{opacity:1;}
/* severity-tinted top hairline */
.ac-card[data-ac-sev]::after{
  content:""; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg, transparent, var(--ac-edge, var(--ac-teal)),
              transparent 82%); opacity:.8;
}
.ac-head{display:flex;align-items:flex-start;gap:.65rem;margin-bottom:.6rem;}
.ac-head .ac-h-ico{color:var(--ac-teal);opacity:.9;margin-top:2px;}
.ac-h-title{font-size:.94rem;font-weight:600;letter-spacing:.005em;color:#fff;line-height:1.25;}
.ac-h-sub{font-family:var(--ac-mono);font-size:.60rem;letter-spacing:.17em;
  color:var(--ac-faint);text-transform:uppercase;margin-top:3px;}

/* section header ---------------------------------------------- */
.ac-sec{display:flex;align-items:center;gap:.6rem;margin:.35rem 0 .1rem;}
.ac-sec-tick{width:3px;height:15px;border-radius:2px;background:var(--ac-teal);
  box-shadow:0 0 9px var(--ac-glow);}
.ac-sec-txt{font-family:var(--ac-mono);font-size:.655rem;letter-spacing:.26em;
  text-transform:uppercase;color:var(--ac-muted);}
.ac-sec-meta{margin-left:auto;font-family:var(--ac-mono);font-size:.575rem;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ac-faint);}

/* key/value readouts ----------------------------------------- */
.ac-kv{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;
  padding:.34rem 0;border-bottom:1px dashed rgba(130,168,200,.11);}
.ac-kv:last-child{border-bottom:0;}
.ac-kv-k{font-family:var(--ac-mono);font-size:.625rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--ac-faint);white-space:nowrap;}
.ac-kv-v{font-family:var(--ac-mono);font-size:.82rem;font-weight:500;color:var(--ac-text);
  text-align:right;letter-spacing:.01em;}

/* status / mission bar ---------------------------------------- */
.ac-bar{
  display:flex;align-items:center;gap:.9rem;flex-wrap:wrap;
  padding:.5rem .85rem;border-radius:var(--ac-r-sm);
  background:linear-gradient(90deg, rgba(0,229,199,.075), rgba(0,229,199,.012) 60%);
  border:1px solid var(--ac-hair); border-left:2px solid var(--ac-teal);
}
.ac-bar .ac-b{font-family:var(--ac-mono);font-size:.635rem;letter-spacing:.13em;
  color:var(--ac-muted);text-transform:uppercase;white-space:nowrap;}
.ac-bar .ac-b b{color:var(--ac-text);font-weight:500;}
.ac-bar .ac-sp{flex:1;}

/* ============ 5 · KPI strip ============ */
.ac-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.85rem;
  width:100%;}
.ac-kpi{
  position:relative; overflow:hidden; border-radius:var(--ac-r);
  padding:.8rem .9rem .72rem;
  background:linear-gradient(158deg, rgba(16,23,38,.80), rgba(12,18,30,.55));
  backdrop-filter:blur(16px) saturate(130%);
  -webkit-backdrop-filter:blur(16px) saturate(130%);
  border:1px solid var(--ac-hair-soft);
  box-shadow:0 16px 40px -28px rgba(0,0,0,.9), inset 0 1px 0 rgba(255,255,255,.05);
  transition:border-color .22s ease, box-shadow .25s ease, transform .22s ease;
}
.ac-kpi::before{ /* accent hairline across the top edge */
  content:""; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg, transparent,
              color-mix(in srgb, var(--kpi) 70%, transparent), transparent 78%);
  opacity:.75;
}
.ac-kpi:hover{
  border-color:color-mix(in srgb, var(--kpi) 42%, transparent);
  transform:translateY(-2px);
  box-shadow:0 22px 48px -26px rgba(0,0,0,.95),
             0 0 0 1px color-mix(in srgb, var(--kpi) 12%, transparent),
             0 12px 42px -20px color-mix(in srgb, var(--kpi) 60%, transparent);
}
.ac-kpi::after{ /* sheen sweep on hover */
  content:""; position:absolute; top:0; left:0; width:42%; height:100%;
  background:linear-gradient(100deg, transparent, rgba(255,255,255,.055), transparent);
  transform:translateX(-130%); pointer-events:none;
}
.ac-kpi:hover::after{animation:ac-sheen .9s ease-out;}
.ac-kpi-idx{position:absolute;top:.55rem;right:.7rem;
  font-family:var(--ac-mono);font-size:.575rem;letter-spacing:.1em;
  color:var(--ac-faint);opacity:.6;}
.ac-kpi-top{display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;}
.ac-kpi-ico{
  width:26px;height:26px;flex:0 0 26px;border-radius:8px;display:grid;place-items:center;
  color:var(--kpi); background:color-mix(in srgb, var(--kpi) 13%, transparent);
  border:1px solid color-mix(in srgb, var(--kpi) 32%, transparent);
  box-shadow:0 0 14px -5px var(--kpi);
}
.ac-kpi-label{font-family:var(--ac-mono);font-size:.585rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--ac-muted);}
.ac-kpi-val{display:flex;align-items:baseline;gap:.22rem;
  font-family:var(--ac-mono);font-weight:600;line-height:1;
  color:#fff; text-shadow:0 0 26px color-mix(in srgb, var(--kpi) 55%, transparent);}
.ac-kpi-num{font-size:2.05rem;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;}
.ac-kpi-unit{font-size:.82rem;color:var(--kpi);letter-spacing:.06em;}
.ac-kpi-foot{display:flex;align-items:flex-end;justify-content:space-between;
  gap:.5rem;margin-top:.55rem;padding-top:.45rem;
  border-top:1px solid rgba(130,168,200,.11);}
.ac-kpi-sub{font-family:var(--ac-mono);font-size:.575rem;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ac-faint);line-height:1.5;}
.ac-kpi-sub b{color:var(--kpi);font-weight:500;}
/* live beacon on a card that is actively changing state */
.ac-kpi-pulse{position:absolute;right:.7rem;bottom:.7rem;width:6px;height:6px;
  border-radius:50%;background:var(--kpi);box-shadow:0 0 8px var(--kpi);}
.ac-kpi-pulse::after{content:"";position:absolute;inset:-3px;border-radius:50%;
  border:1px solid var(--kpi);opacity:.5;animation:ac-pulse 2.2s ease-out infinite;}

/* mini stat row (inspect panels) ----------------------------- */
.ac-minis{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
  gap:.6rem;width:100%;}
.ac-mini{
  position:relative;border-radius:var(--ac-r-sm);padding:.5rem .65rem;
  background:rgba(15,22,36,.5);border:1px solid var(--ac-hair-soft);
  border-left:2px solid var(--kpi, var(--ac-teal));
}
.ac-mini .ac-m-k{font-family:var(--ac-mono);font-size:.545rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--ac-faint);}
.ac-mini .ac-m-v{font-family:var(--ac-mono);font-size:1.02rem;font-weight:600;
  color:var(--kpi,#fff);margin-top:3px;letter-spacing:-.01em;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* ============ 6 · pills / badges / legend ============ */
.ac-pill{
  display:inline-flex;align-items:center;gap:.34rem;
  padding:.16rem .55rem .2rem;border-radius:999px;
  font-family:var(--ac-mono);font-size:.60rem;font-weight:600;letter-spacing:.15em;
  text-transform:uppercase;white-space:nowrap;
  color:var(--pill,#E6EDF6);
  background:color-mix(in srgb, var(--pill,#888) 15%, transparent);
  border:1px solid color-mix(in srgb, var(--pill,#888) 45%, transparent);
  box-shadow:0 0 14px -5px var(--pill,#888), inset 0 0 12px -9px var(--pill,#888);
}
.ac-pill i{width:5px;height:5px;border-radius:50%;background:currentColor;
  box-shadow:0 0 6px currentColor;}
.ac-tag{
  display:inline-flex;align-items:center;gap:.3rem;padding:.12rem .45rem;
  border-radius:6px;font-family:var(--ac-mono);font-size:.575rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ac-muted);
  background:rgba(130,168,200,.07);border:1px solid var(--ac-hair-soft);
}
.ac-legend{display:flex;align-items:center;gap:.85rem;flex-wrap:wrap;
  font-family:var(--ac-mono);font-size:.575rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ac-faint);}
.ac-legend span{display:inline-flex;align-items:center;gap:.34rem;}
.ac-legend em{width:14px;height:0;border-top:1.5px solid currentColor;font-style:normal;}
.ac-legend em.dash{border-top-style:dashed;}
.ac-legend em.dot{width:6px;height:6px;border:0;border-radius:1px;
  transform:rotate(45deg);background:currentColor;}

/* caption used under charts ---------------------------------- */
.ac-cap{font-family:var(--ac-mono);font-size:.6rem;line-height:1.7;
  letter-spacing:.13em;color:var(--ac-faint);text-transform:uppercase;
  padding:.55rem .25rem 0;display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;}

/* ============ 7 · meters ============ */
.ac-meter{display:flex;align-items:center;gap:.7rem;width:100%;
  padding:.36rem 0;}
.ac-meter-lab{flex:1;min-width:0;}
.ac-meter-k{font-family:var(--ac-mono);font-size:.60rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--ac-muted);}
.ac-meter-track{
  position:relative;height:5px;border-radius:5px;margin-top:.34rem;overflow:hidden;
  background:
    repeating-linear-gradient(90deg, rgba(255,255,255,.05) 0 1px, transparent 1px 9px),
    rgba(130,168,200,.10);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.03);
}
.ac-meter-fill{position:absolute;inset:0 auto 0 0;border-radius:5px;
  background:linear-gradient(90deg,
    color-mix(in srgb, var(--mc,#00E5C7) 55%, transparent), var(--mc,#00E5C7));
  box-shadow:0 0 12px -1px var(--mc,#00E5C7);
  animation:ac-meter-in .9s cubic-bezier(.2,.8,.2,1) both;}
.ac-meter-cap{position:absolute;top:-3px;bottom:-3px;width:1px;
  background:rgba(255,255,255,.28);}
.ac-meter-v{font-family:var(--ac-mono);font-size:.80rem;font-weight:600;
  color:var(--mc,#fff);min-width:52px;text-align:right;
  font-variant-numeric:tabular-nums;}

/* ============ 8 · buttons ============ */
/* Streamlit 1.64 exposes the button kind through the testid:
   [data-testid="stBaseButton-<kind>"]. Confirm is type="primary",
   Reject is the default type="secondary". */
.stButton > button, .stDownloadButton > button, .stLinkButton > button{
  width:100%;border-radius:var(--ac-r-sm) !important;
  font-family:var(--ac-mono);font-size:.635rem;letter-spacing:.19em;
  text-transform:uppercase;font-weight:500 !important;
  padding:.62rem .9rem !important;min-height:40px;
  border:1px solid var(--ac-hair-soft) !important;
  background:rgba(18,26,42,.72) !important;color:var(--ac-text) !important;
  box-shadow:none !important;
  transition:border-color .18s ease, box-shadow .22s ease,
             background .18s ease, color .18s ease, transform .12s ease;
}
.stButton > button:hover{
  border-color:rgba(0,229,199,.45) !important;color:#fff !important;
  background:rgba(0,229,199,.10) !important;
  box-shadow:0 0 22px -6px var(--ac-glow) !important;
}
.stButton > button:active{transform:translateY(1px);}
.stButton > button:focus-visible{outline:1px solid var(--ac-teal);outline-offset:2px;}

/* teal Confirm — the promote-to-live-feed action */
.stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg, rgba(0,229,199,.92), rgba(14,158,140,.88)) !important;
  border-color:rgba(0,229,199,.78) !important;color:#04211D !important;
  font-weight:700 !important;
  box-shadow:0 8px 26px -10px rgba(0,229,199,.75),
             inset 0 1px 0 rgba(255,255,255,.28) !important;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover{
  background:linear-gradient(135deg, #12F2D6, #0FB6A2) !important;
  box-shadow:0 12px 34px -8px rgba(0,229,199,.85) !important;
}

/* muted red Reject — deliberately desaturated; rejection is a logged,
   reversible editorial act, not an emergency action */
.stButton > button[data-testid="stBaseButton-secondary"]{
  background:rgba(255,77,77,.07) !important;
  border-color:rgba(255,77,77,.32) !important;color:#FF9E9E !important;
}
.stButton > button[data-testid="stBaseButton-secondary"]:hover{
  background:rgba(255,77,77,.16) !important;
  border-color:rgba(255,77,77,.6) !important;color:#FFD0D0 !important;
  box-shadow:0 0 22px -8px rgba(255,77,77,.7) !important;
}

/* ============ 9 · inputs ============ */
[data-baseweb="input"], [data-baseweb="select"] > div{
  background:rgba(10,15,25,.72) !important;
  border:1px solid var(--ac-hair-soft) !important;
  border-radius:var(--ac-r-sm) !important;
  color:var(--ac-text) !important;
  transition:border-color .18s ease, box-shadow .2s ease;
}
[data-baseweb="input"]:hover, [data-baseweb="select"] > div:hover{
  border-color:rgba(0,229,199,.32) !important;}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within{
  border-color:rgba(0,229,199,.6) !important;
  box-shadow:0 0 0 1px rgba(0,229,199,.25), 0 0 22px -10px var(--ac-glow) !important;
}
[data-baseweb="input"] input, [data-baseweb="select"] input,
[data-baseweb="input"] textarea{
  font-family:var(--ac-mono) !important;font-size:.80rem !important;
  color:var(--ac-text) !important; caret-color:var(--ac-teal) !important;}
[data-baseweb="select"] div{font-family:var(--ac-mono) !important;
  font-size:.80rem !important;letter-spacing:.03em;}
[data-testid="stSelectbox"] label, [data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label, [data-testid="stSlider"] label,
[data-testid="stTextArea"] label{
  font-family:var(--ac-mono) !important;font-size:.60rem !important;
  letter-spacing:.19em !important;text-transform:uppercase;
  color:var(--ac-faint) !important;font-weight:500 !important;}
[data-testid="stNumberInput"] button{background:rgba(0,229,199,.07) !important;
  border-color:var(--ac-hair-soft) !important;color:var(--ac-teal) !important;}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:var(--ac-teal) !important;border:2px solid #0A0E17 !important;
  box-shadow:0 0 0 1px rgba(0,229,199,.7), 0 0 16px -2px var(--ac-glow) !important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{
  font-family:var(--ac-mono) !important;font-size:.60rem !important;
  color:var(--ac-faint) !important;}
[data-testid="stCheckbox"] label p,[data-testid="stRadio"] label p{color:inherit;}

/* ============ 10 · dataframe ============ */
[data-testid="stDataFrame"]{border-radius:var(--ac-r) !important;overflow:hidden;
  border:1px solid var(--ac-hair-soft);
  background:linear-gradient(158deg, rgba(15,22,36,.55), rgba(11,17,29,.4)) !important;
  box-shadow:0 16px 40px -30px rgba(0,0,0,.9);}
[data-testid="stDataFrame"] [role="columnheader"]{
  background:rgba(0,229,199,.07) !important;
  border-bottom:1px solid var(--ac-hair) !important;}
[data-testid="stDataFrame"] [role="gridcell"]{
  font-family:var(--ac-mono) !important;font-size:.72rem !important;
  background:transparent !important;}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{
  background:rgba(11,16,27,.45) !important;}
[data-testid="stDataFrame"] [role="progressbar"]{
  background:rgba(130,168,200,.12) !important;}

/* ============ 11 · plotly + geo HUD ============ */
div[data-testid="stPlotlyChart"]{border-radius:var(--ac-r);overflow:hidden;}
.js-plotly-plot .plotly .modebar{display:none !important;}
.js-plotly-plot, .plot-container{background:transparent !important;}

/* continuous-monitoring overlays, scoped by st.container(key="ac_geo").
   Streamlit keys a container with a `st-key-<key>` class, which is what
   lets the map panel be styled as a glass card and carry the HUD. */
.st-key-ac_geo{
  position:relative; overflow:hidden; padding:.45rem;
  border-radius:var(--ac-r);
  border:1px solid var(--ac-hair-soft);
  background:linear-gradient(158deg, var(--ac-glass), var(--ac-glass-2));
  backdrop-filter:blur(16px) saturate(125%);
  -webkit-backdrop-filter:blur(16px) saturate(125%);
  box-shadow:0 18px 44px -30px rgba(0,0,0,.95),
             inset 0 1px 0 rgba(255,255,255,.05);
  transition:border-color .22s ease, box-shadow .25s ease;
}
.st-key-ac_geo:hover{
  border-color:rgba(0,229,199,.28);
  box-shadow:0 22px 50px -28px rgba(0,0,0,.95),
             0 10px 44px -24px var(--ac-glow);
}
.st-key-ac_geo > div{position:relative;}
/* targeting reticle in the panel corners */
.ac-frame{position:absolute;inset:5px;pointer-events:none;z-index:4;}
.ac-frame i{position:absolute;width:15px;height:15px;
  border:1px solid rgba(0,229,199,.5);}
.ac-frame i:nth-child(1){top:0;left:0;border-right:0;border-bottom:0;}
.ac-frame i:nth-child(2){top:0;right:0;border-left:0;border-bottom:0;}
.ac-frame i:nth-child(3){bottom:0;left:0;border-right:0;border-top:0;}
.ac-frame i:nth-child(4){bottom:0;right:0;border-left:0;border-top:0;}
/* HUD strip above the chart */
.ac-hud{
  display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;
  padding:.15rem .3rem .5rem;
  font-family:var(--ac-mono);font-size:.575rem;letter-spacing:.19em;
  text-transform:uppercase;color:var(--ac-faint);
}
.ac-hud b{color:var(--ac-muted);font-weight:500;}
.ac-hud .ac-sp{flex:1;}
.ac-hud .ac-live{margin:0;padding:.18rem .45rem;}
.ac-hud .ac-live span{font-size:.575rem;letter-spacing:.16em;}
/* radar sweep ring, bottom-right */
.st-key-ac_geo::before{
  content:""; position:absolute; right:-58px; bottom:-58px;
  width:clamp(180px,24vw,320px);height:clamp(180px,24vw,320px);
  border-radius:50%;pointer-events:none;z-index:2;opacity:.55;
  background:conic-gradient(from 0deg,
      rgba(0,229,199,0) 0deg, rgba(0,229,199,0) 296deg,
      rgba(0,229,199,.30) 344deg, rgba(0,229,199,.05) 358deg,
      rgba(0,229,199,0) 360deg);
  -webkit-mask:radial-gradient(circle, transparent 0 60%, #000 62%, #000 96%,
              transparent 98%);
  mask:radial-gradient(circle, transparent 0 60%, #000 62%, #000 96%,
              transparent 98%);
  animation:ac-sweep 7.5s linear infinite;
}
/* vertical scan line */
.st-key-ac_geo::after{
  content:""; position:absolute; left:0; right:0; top:0; height:34%;
  pointer-events:none; z-index:3; opacity:.8;
  background:linear-gradient(180deg, rgba(0,229,199,0) 0%,
              rgba(0,229,199,.045) 55%, rgba(0,229,199,.15) 88%,
              rgba(0,229,199,.02) 100%);
  animation:ac-scan 7.5s cubic-bezier(.45,0,.55,1) infinite;
}

/* ============ 12 · review decision cards ============ */
/* Streamlit keys each card container as st-key-ac_dec_<n>; the attribute
   selector catches every one of them with a single rule. */
div[class*="st-key-ac_dec"]{
  position:relative; border-radius:var(--ac-r); padding:.15rem;
  border:1px solid var(--ac-hair-soft);
  background:linear-gradient(158deg, var(--ac-glass), var(--ac-glass-2));
  backdrop-filter:blur(16px) saturate(125%);
  -webkit-backdrop-filter:blur(16px) saturate(125%);
  box-shadow:0 18px 44px -30px rgba(0,0,0,.95),
             inset 0 1px 0 rgba(255,255,255,.05);
  transition:border-color .22s ease, box-shadow .25s ease;
}
div[class*="st-key-ac_dec"]:hover{
  border-color:rgba(0,229,199,.26);
  box-shadow:0 22px 50px -28px rgba(0,0,0,.95),
             0 10px 44px -26px var(--ac-glow);
}
div[class*="st-key-ac_dec"] [data-testid="stHorizontalBlock"]
  > [data-testid="stColumn"]:last-child{
  border-left:1px solid var(--ac-hair-soft); padding-left:1rem;
}
.ac-dec-info{border-left:2px solid var(--dec, var(--ac-amber));
  padding-left:.85rem;}
.ac-dec-top{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;}
.ac-dec-id{margin-left:auto;font-family:var(--ac-mono);font-size:.62rem;
  letter-spacing:.07em;color:var(--ac-faint);}
.ac-dec-coord{font-family:var(--ac-mono);font-size:1.02rem;font-weight:500;
  color:#fff;letter-spacing:.02em;margin:.5rem 0 .2rem;
  text-shadow:0 0 22px color-mix(in srgb, var(--dec,#00E5C7) 45%, transparent);}
.ac-dec-coord small{color:var(--ac-faint);font-size:.58rem;letter-spacing:.2em;
  margin-right:.45rem;font-weight:400;}
/* grid of small readouts under the coordinate */
.ac-dec-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(126px,1fr));
  gap:.1rem 1.1rem;margin-top:.4rem;}
.ac-dec-grid div{display:flex;justify-content:space-between;gap:.6rem;
  padding:.22rem 0;border-bottom:1px dashed rgba(130,168,200,.11);}
.ac-dec-grid em{font-family:var(--ac-mono);font-size:.6rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ac-faint);font-style:normal;white-space:nowrap;}
.ac-dec-grid b{font-family:var(--ac-mono);font-size:.78rem;font-weight:500;
  color:var(--ac-text);white-space:nowrap;}
.ac-dec-rail-top{display:flex;align-items:center;gap:.85rem;margin-bottom:.35rem;}
.ac-dec-rail-top .ac-dt{font-family:var(--ac-mono);font-size:.575rem;
  letter-spacing:.18em;text-transform:uppercase;color:var(--ac-faint);line-height:1.6;}
.ac-dec-rail-top .ac-dt b{display:block;color:var(--ac-text);font-size:.72rem;
  letter-spacing:.1em;font-weight:500;}

/* ============ 13 · st messages ============ */
[data-testid="stAlert"]{
  border-radius:var(--ac-r-sm) !important;
  background:rgba(15,22,36,.66) !important;
  backdrop-filter:blur(12px);
  border:1px solid var(--ac-hair-soft) !important;
  border-left:2px solid var(--al,#00E5C7) !important;
  color:var(--ac-text) !important;font-size:.80rem !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{color:var(--ac-text);}
[data-testid="stToast"]{background:rgba(13,20,33,.96) !important;
  border:1px solid var(--ac-hair) !important;}

/* ============ 14 · keyframes ============ */
@keyframes ac-spin{to{transform:rotate(360deg);}}
@keyframes ac-sheen{from{transform:translateX(-130%);}to{transform:translateX(330%);}}
@keyframes ac-meter-in{from{transform:scaleX(0);transform-origin:left;}}
@keyframes ac-pulse{
  0%{transform:scale(.7);opacity:.75;}
  70%{transform:scale(2.1);opacity:0;}
  100%{transform:scale(2.1);opacity:0;}}
@keyframes ac-blink{0%,100%{opacity:1;}50%{opacity:.32;}}
@keyframes ac-sweep{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
@keyframes ac-scan{
  0%{transform:translateY(-105%);} 55%{transform:translateY(300%);}
  100%{transform:translateY(300%);}
}

/* ============ 15 · responsive ============ */
/* 1080p ops-room wall: keep the numerals and panel padding generous. */
@media (min-width:1680px){
  [data-testid="stMainBlockContainer"]{padding:.6rem 2.1rem 2.8rem !important;}
  [data-testid="stVerticalBlock"]{gap:.85rem;}
  .ac-kpi{padding:.95rem 1.05rem .85rem;}
  .ac-kpi-num{font-size:2.35rem;}
  .ac-card{padding:1.1rem 1.25rem;}
  :root{--ac-r:16px;}
}
@media (max-width:1400px){
  [data-testid="stMainBlockContainer"]{padding-left:1.15rem !important;
    padding-right:1.15rem !important;}
}
@media (max-width:1200px){
  .ac-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}
  div[class*="st-key-ac_dec"] [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:last-child{
    border-left:0;border-top:1px solid var(--ac-hair-soft);
    padding-left:0;padding-top:.8rem;}
  [data-testid="stSidebar"],[data-testid="stSidebarContainer"]{
    min-width:236px;max-width:236px;}
}
@media (max-width:820px){
  .ac-kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:.6rem;}
  .ac-kpi-num{font-size:1.7rem;}
  .st-key-ac_geo::before{display:none;}
  [data-testid="stMainBlockContainer"]{padding:.3rem .7rem 2rem !important;}
}
@media (max-width:560px){
  .ac-kpis{grid-template-columns:minmax(0,1fr);}
  .ac-dec-coord{font-size:.9rem;}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms !important;
    animation-iteration-count:1 !important;transition-duration:.001ms !important;}
}
"""


def inject_css(nav: bool = True) -> None:
    """Install the Amplicast stylesheet. Call once, before any widget.

    Streamlit 1.64 sanitises injected HTML with DOMPurify, which strips
    `<style>` elements.  We therefore inject the rules through a `<script>`
    (allowed when ``unsafe_allow_javascript`` is on) that builds the
    `<style>` at runtime — so the visual identity actually reaches the
    browser.
    """
    import streamlit as st
    block = _CSS + ("\n" + _nav_css() if nav else "")
    # keep the CSS out of the JS template so the CSS is not mangled
    payload = (
        "var s=document.createElement('style');"
        "s.textContent=" + repr(block) + ";"
        "document.head.appendChild(s);"
    )
    st.html(f"<script>(function(){{{payload}}})()</script>",
            unsafe_allow_javascript=True)


# --------------------------------------------------------------------------
# 5. JavaScript micro-interactions
# --------------------------------------------------------------------------
# Count-up on KPI numerals + the UTC ops clock. Guarded by a data flag so a
# Streamlit re-render always replays the animation exactly once per node.
_JS = """
<script>
(function(){
  var run = function(){
    var reduce = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var fmt = function(v, dec, grp){
      var s = v.toFixed(dec);
      if (!grp) return s;
      var p = s.split('.');
      p[0] = p[0].replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
      return p.join('.');
    };
    document.querySelectorAll('[data-ac-count]').forEach(function(el){
      if (el.getAttribute('data-ac-done') === '1') return;
      el.setAttribute('data-ac-done', '1');
      var to = parseFloat(el.getAttribute('data-ac-count'));
      if (isNaN(to)) return;
      var dec = parseInt(el.getAttribute('data-ac-dec') || '0', 10);
      var grp = el.getAttribute('data-ac-group') === '1';
      if (reduce || to <= 0) { el.textContent = fmt(to, dec, grp); return; }
      var dur = 1150, t0 = null;
      var step = function(ts){
        if (t0 === null) t0 = ts;
        var p = Math.min(1, (ts - t0) / dur);
        var e = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(to * e, dec, grp);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    var clock = document.getElementById('ac-ops-clock');
    if (clock && !clock.getAttribute('data-ac-done')) {
      clock.setAttribute('data-ac-done', '1');
      var pad = function(n){ return n < 10 ? '0' + n : '' + n; };
      var tick = function(){
        var d = new Date();
        clock.textContent = pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) +
          ':' + pad(d.getUTCSeconds()) + 'Z';
      };
      tick(); setInterval(tick, 1000);
    }
  };
  if (document.readyState === 'complete') { run(); }
  else { window.addEventListener('load', run); }
  requestAnimationFrame(run);
})();
</script>
"""


# --------------------------------------------------------------------------
# 6. HTML component helpers
# --------------------------------------------------------------------------
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render(markup: str, js: bool = False) -> None:
    """Emit raw HTML (optionally with the motion script) into the page."""
    import streamlit as st
    st.html(markup + (_JS if js else ""), unsafe_allow_javascript=True)


def kpi_strip(cards: list[dict]) -> None:
    """Top KPI strip. Each card: index, icon, label, count-up value, sub, spark.

    ``cards`` entries: {label, value, unit, sub, accent, icon, spark, pulse}
    """
    cells = []
    for i, c in enumerate(cards, start=1):
        accent = c.get("accent", TEAL)
        unit = (f"<span class='ac-kpi-unit'>{_esc(c['unit'])}</span>"
                if c.get("unit") else "")
        dec = c.get("dec", 0)
        value = float(c["value"])
        grp = " data-ac-group='1'" if abs(value) >= 1000 else ""
        num = (f"<span class='ac-kpi-num' data-ac-count='{value:g}' "
               f"data-ac-dec='{dec}'{grp}>0</span>")
        pulse = "<i class='ac-kpi-pulse'></i>" if c.get("pulse") else ""
        cells.append(
            f"<div class='ac-kpi' style='--kpi:{accent}'>"
            f"  <span class='ac-kpi-idx'>{i:02d}</span>"
            f"  <div class='ac-kpi-top'>"
            f"    <span class='ac-kpi-ico'>{icon(c.get('icon','gauge'),14)}</span>"
            f"    <span class='ac-kpi-label'>{_esc(c['label'])}</span>"
            f"  </div>"
            f"  <div class='ac-kpi-val'>{num}{unit}</div>"
            f"  <div class='ac-kpi-foot'>"
            f"    <span class='ac-kpi-sub'>{c.get('sub','')}</span>"
            f"    {c.get('spark','')}"
            f"  </div>{pulse}"
            f"</div>"
        )
    render(f"<div class='ac-kpis'>{''.join(cells)}</div>", js=True)


def minis(items: list[tuple]) -> None:
    """Compact stat row: [(label, value, accent), ...]."""
    cells = "".join(
        f"<div class='ac-mini' style='--kpi:{a}'>"
        f"<div class='ac-m-k'>{_esc(k)}</div>"
        f"<div class='ac-m-v'>{_esc(v)}</div></div>"
        for k, v, a in items
    )
    render(f"<div class='ac-minis'>{cells}</div>")


def section(text: str, meta: str = "") -> None:
    extra = f"<span class='ac-sec-meta'>{_esc(meta)}</span>" if meta else ""
    render(f"<div class='ac-sec'><span class='ac-sec-tick'></span>"
           f"<span class='ac-sec-txt'>{_esc(text)}</span>{extra}</div>")


def card(inner: str, sev: str | None = None, cls: str = "") -> str:
    """Compose a glass card; returns markup (caller decides how to place it)."""
    if sev:
        return (f"<div class='ac-card {cls}' data-ac-sev='1' "
                f"style='--ac-edge:{SEV_COLOR.get(sev, SLATE)}'>{inner}</div>")
    return f"<div class='ac-card {cls}'>{inner}</div>"


def card_head(title: str, sub: str = "", ico: str = "gauge") -> str:
    return (f"<div class='ac-head'><span class='ac-h-ico'>{icon(ico,15)}</span>"
            f"<div><div class='ac-h-title'>{_esc(title)}</div>"
            f"{f'<div class=ac-h-sub>{_esc(sub)}</div>' if sub else ''}</div></div>")


def pill(text: str, color: str, dot: bool = True) -> str:
    return (f"<span class='ac-pill' style='--pill:{color}'>"
            f"{'<i></i>' if dot else ''}{_esc(text)}</span>")


def sev_pill(sev: str) -> str:
    return pill((sev or "unknown").upper(), SEV_COLOR.get(sev, SLATE))


def tag(text: str, ico: str = "") -> str:
    return f"<span class='ac-tag'>{icon(ico, 11) if ico else ''}{_esc(text)}</span>"


def legend(items: list[tuple]) -> str:
    """Chart key: [(color, label), ...] or [(color, label, kind), ...].

    ``kind`` is one of ``line`` (solid rule), ``dash`` (dashed rule) or
    ``dot`` (diamond core) and mirrors how the series is drawn on the chart.
    """
    marks = {"line": "<em></em>", "dash": "<em class='dash'></em>",
             "dot": "<em class='dot'></em>"}
    parts = []
    for item in items:
        color, label = item[0], item[1]
        kind = item[2] if len(item) > 2 else "line"
        parts.append(f"<span style='color:{color}'>{marks.get(kind, marks['line'])}"
                     f"{_esc(label)}</span>")
    return f"<div class='ac-legend'>{''.join(parts)}</div>"


def caption(text: str) -> str:
    return f"<div class='ac-cap'>{text}</div>"


def note(text: str, tone: str = "amber") -> str:
    """Operational caveat strip — the trust / human-oversight voice."""
    edge = {"amber": AMBER, "teal": TEAL, "red": RED}.get(tone, AMBER)
    return (f"<div class='ac-note' style='border-left-color:{edge}88'>"
            f"{text}</div>")


def kvs(pairs: list[tuple]) -> str:
    """Monospace key/value readout rows: [(key, value_html), ...]."""
    return "".join(
        f"<div class='ac-kv'><span class='ac-kv-k'>{_esc(k)}</span>"
        f"<span class='ac-kv-v'>{v}</span></div>"
        for k, v in pairs
    )


def readout_grid(pairs: list[tuple]) -> str:
    """Same readouts as :func:`kvs`, laid out as a responsive 2-up grid.

    Used inside decision cards, where a stacked list would push the action
    rail below the fold on a laptop screen.
    """
    return ("<div class='ac-dec-grid'>" + "".join(
        f"<div><em>{_esc(k)}</em><b>{v}</b></div>" for k, v in pairs) + "</div>")


def meter(label: str, value: float, color: str | None = None, fmt: str = "{:.3f}",
          suffix: str = "", ticks: bool = True) -> str:
    """Thin bar meter with a glowing fill and tabular-numeric readout."""
    v = max(0.0, min(1.0, float(value)))
    c = color or _conf_color(v)
    grid = ("repeating-linear-gradient(90deg, rgba(255,255,255,.06) 0 1px, "
            "transparent 1px 10%)" if ticks else "none")
    return (f"<div class='ac-meter' style='--mc:{c}'>"
            f"  <div class='ac-meter-lab'>"
            f"    <div class='ac-meter-k'>{_esc(label)}</div>"
            f"    <div class='ac-meter-track' style='background-image:{grid}'>"
            f"      <span class='ac-meter-fill' style='width:{v * 100:.1f}%'></span>"
            f"    </div>"
            f"  </div>"
            f"  <div class='ac-meter-v'>{fmt.format(v)}{suffix}</div>"
            f"</div>")


def _conf_color(v: float) -> str:
    if v >= 0.66:
        return TEAL
    if v >= 0.4:
        return AMBER
    return RED


def radial(value: float, color: str, size: int = 62, label: str = "",
           sub: str = "", width: float = 4.0) -> str:
    """Thin radial confidence gauge rendered as SVG."""
    v = max(0.0, min(1.0, float(value)))
    r = size / 2 - 6
    circ = 2 * math.pi * r
    cap = ""
    if label:
        cap = ("<text x='50%' y='86%' text-anchor='middle' fill='#61748C' "
               f"font-family='{FONT_MONO}' font-size='6' "
               f"letter-spacing='1'>{_esc(label)}</text>")
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}' "
        f"style='flex:0 0 {size}px;filter:drop-shadow(0 0 8px {color}55)'>"
        f"<circle cx='{size / 2}' cy='{size / 2}' r='{r:.2f}' fill='none' "
        f"stroke='rgba(130,168,200,.16)' stroke-width='{width}'/>"
        f"<circle cx='{size / 2}' cy='{size / 2}' r='{r:.2f}' fill='none' "
        f"stroke='{color}' stroke-width='{width}' stroke-linecap='round' "
        f"stroke-dasharray='{circ:.2f}' stroke-dashoffset='{circ * (1 - v):.2f}' "
        f"transform='rotate(-90 {size / 2} {size / 2})'/>"
        f"<text x='50%' y='50%' dy='.35em' text-anchor='middle' fill='{color}' "
        f"font-family='{FONT_MONO}' font-size='13' font-weight='600'>"
        f"{v * 100:.0f}</text>{cap}</svg>"
    ) + (f"<div style='font-family:{FONT_MONO};font-size:.575rem;letter-"
         f"spacing:.14em;text-transform:uppercase;color:{color};margin-top:.15rem'>"
         f"{_esc(sub)}</div>" if sub else "")


# --------------------------------------------------------------------------
# 7. Sparklines
# --------------------------------------------------------------------------
def _scale(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def sparkline(values, color: str = TEAL, kind: str = "line",
              w: int = 74, h: int = 24) -> str:
    """Tiny SVG trend glyph for a KPI card. ``values`` may contain ``None``."""
    vals = [0.0 if v is None else float(v) for v in values] if values else []
    if not vals:
        return ""
    if kind == "bar":
        n = len(vals)
        bw = max(1.5, (w - (n - 1) * 1.4) / n)
        norm = _scale(vals)
        bars = "".join(
            f"<rect x='{i * (bw + 1.4):.2f}' y='{h - max(1.6, v * (h - 2)):.2f}' "
            f"width='{bw:.2f}' height='{max(1.6, v * (h - 2)):.2f}' rx='1' "
            f"fill='{color}' opacity='{0.34 + 0.62 * v:.2f}'/>"
            for i, v in enumerate(norm))
        return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
                f"style='flex:0 0 auto;overflow:visible'>{bars}</svg>")

    norm = _scale(vals)
    pts = [(i * (w / max(1, len(vals) - 1)), h - 2 - v * (h - 4))
           for i, v in enumerate(norm)]
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    area = f"{line} {w},{h} 0,{h}"
    gid = f"acg{abs(hash((tuple(vals), color))) % 99999}"
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
            f"style='flex:0 0 auto;overflow:visible'>"
            f"<defs><linearGradient id='{gid}' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0' stop-color='{color}' stop-opacity='.34'/>"
            f"<stop offset='1' stop-color='{color}' stop-opacity='0'/>"
            f"</linearGradient></defs>"
            f"<polygon points='{area}' fill='url(#{gid})'/>"
            f"<polyline points='{line}' fill='none' stroke='{color}' "
            f"stroke-width='1.4' stroke-linejoin='round' stroke-linecap='round'/>"
            f"<circle cx='{pts[-1][0]:.2f}' cy='{pts[-1][1]:.2f}' r='1.9' "
            f"fill='{color}'/></svg>")


# --------------------------------------------------------------------------
# 8. Sidebar chrome
# --------------------------------------------------------------------------
def brand_mark() -> None:
    render(
        "<div class='ac-brand'>"
        "  <div class='ac-mark'>" + icon("satellite", 20) + "</div>"
        "  <div><div class='ac-name'>AMPLICAST</div>"
        "  <div class='ac-tag'>WEATHER INTELLIGENCE</div></div>"
        "</div>"
    )


def nav_eyebrow(text: str = "Command views") -> None:
    render(f"<div class='ac-nav-eyebrow'>{_esc(text)}</div>")


def live_block(label: str, tone: str = "") -> None:
    render(f"<div class='ac-live {tone}'><span class='ac-dot'></span>"
           f"<span>{_esc(label)}</span></div>")


def side_note(text: str) -> None:
    render(f"<div class='ac-note'>{_esc(text)}</div>")


def sys_readout(items: list[tuple]) -> None:
    """Sidebar system-stack readout: [(label, value), ...]."""
    rows = "".join(f"<div class='ac-sys-row'><i></i><b>{_esc(k)}</b>"
                   f"<span>{_esc(v)}</span></div>" for k, v in items)
    render(f"<div class='ac-sys'>{rows}</div>")


def ops_bar(items: list[tuple], status: str = "", tone: str = "") -> None:
    """Mission strip: [('key','value'), ...] plus an optional status label."""
    cells = "".join(f"<span class='ac-b'>{_esc(k)} <b>{_esc(v)}</b></span>"
                    for k, v in items)
    lead = ""
    if status:
        lead = (f"<div class='ac-live {tone}' style='margin:0'>"
                f"<span class='ac-dot'></span><span>{_esc(status)}</span></div>")
    render(f"<div class='ac-bar'>{lead}{cells}"
           f"<span class='ac-sp'></span>"
           f"<span class='ac-b'>UTC <b id='ac-ops-clock'>--:--:--</b></span></div>",
           js=True)


def page_header(eyebrow: str, title: str, sub: str = "",
                chips: list[tuple] | None = None) -> None:
    """Replace the default st.title with an ops-room page header."""
    right = "".join(f"<span class='ac-pill' style='--pill:{c}'>{_esc(t)}</span>"
                    for t, c in (chips or []))
    render(
        f"<div style='display:flex;align-items:flex-end;justify-content:"
        f"space-between;gap:1rem;flex-wrap:wrap;margin:.15rem 0 .35rem'>"
        f"  <div>"
        f"    <div class='ac-sec' style='margin:0 0 .3rem'>"
        f"      <span class='ac-sec-tick'></span>"
        f"      <span class='ac-sec-txt'>{_esc(eyebrow)}</span></div>"
        f"    <div style='font-size:1.42rem;font-weight:600;color:#fff;"
        f"letter-spacing:-.015em;line-height:1.1'>{_esc(title)}</div>"
        f"    <div style='font-family:{FONT_MONO};font-size:.615rem;"
        f"letter-spacing:.15em;color:#61748C;margin-top:.35rem;"
        f"text-transform:uppercase'>{_esc(sub)}</div>"
        f"  </div>"
        f"  <div style='display:flex;gap:.4rem;flex-wrap:wrap'>{right}</div>"
        f"</div>"
    )


# --------------------------------------------------------------------------
# 9. Geo panel chrome
# --------------------------------------------------------------------------
def geo_hud(left: str, right: str = "", live: tuple | None = None) -> None:
    """Header strip inside a keyed geo container.

    ``live`` is an optional ``(label, tone)`` pair rendered as a pulsing chip.
    """
    chip = ""
    if live:
        chip = (f"<div class='ac-live {live[1]}'>"
                f"<span class='ac-dot'></span><span>{_esc(live[0])}</span></div>")
    render(f"<div class='ac-hud'><b>{_esc(left)}</b>"
           f"<span class='ac-sp'></span>{_esc(right)}{chip}</div>")


def geo_frame() -> None:
    """Corner targeting reticle overlay for a keyed geo container."""
    render("<div class='ac-frame'><i></i><i></i><i></i><i></i></div>")
