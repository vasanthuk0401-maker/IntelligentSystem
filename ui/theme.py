"""Visual theme and HTML components for the Streamlit dashboard.

All dynamic text is HTML-escaped before it is placed in markup, because job
postings and CVs are untrusted input. Markup is emitted on a single line so
Streamlit's markdown renderer never mistakes indented HTML for a code block.
"""
from __future__ import annotations

import html
from typing import Iterable, List, Optional, Sequence, Tuple

import streamlit as st

# Palette
INDIGO, VIOLET, PINK, AMBER, EMERALD, SKY, ROSE = (
    "#6366F1", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#0EA5E9", "#F43F5E")
CHIP_CYCLE = ["indigo", "violet", "pink", "sky", "emerald", "amber"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background:
  radial-gradient(1200px 500px at 100% -10%, rgba(236,72,153,.10), transparent 60%),
  radial-gradient(900px 500px at -10% 10%, rgba(99,102,241,.12), transparent 60%), #F7F6FD; }
.block-container { padding-top: 3.2rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -.02em; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1E1B4B 0%, #312E81 55%, #4C1D95 100%); }
[data-testid="stSidebar"] * { color: #EEF2FF; }
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] input { color: #1E1B4B !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
.sb-brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem; }
.sb-logo { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.35rem;
  background: linear-gradient(135deg, #F472B6, #A78BFA 50%, #60A5FA); box-shadow: 0 6px 18px rgba(236,72,153,.35); }
.sb-title { font-weight:800; font-size:1.08rem; line-height:1.15; }
.sb-sub { font-size:.74rem; opacity:.7; }
.sb-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.35rem .7rem; border-radius:999px;
  font-size:.78rem; font-weight:600; margin:.2rem 0 .6rem; }
.sb-pill.on { background: rgba(16,185,129,.18); border:1px solid rgba(16,185,129,.5); }
.sb-pill.off { background: rgba(245,158,11,.16); border:1px solid rgba(245,158,11,.5); }
.sb-flow { font-size:.74rem; line-height:1.7; opacity:.85; }
.sb-flow b { color:#F9A8D4; }

/* Tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; background: rgba(255,255,255,.7); padding:.35rem; border-radius:14px;
  box-shadow: 0 1px 2px rgba(30,27,75,.06); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:.5rem 1rem !important; font-weight:600; height:auto; }
.stTabs [data-baseweb="tab"] p { font-size:.95rem; font-weight:700; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366F1, #8B5CF6 60%, #EC4899); color:#fff !important; }
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, .stLinkButton > a {
  border-radius:12px; font-weight:600; border:1px solid #DDD6FE; transition: transform .08s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 16px rgba(124,58,237,.18); border-color:#A78BFA; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6366F1, #8B5CF6 55%, #EC4899); border:none; color:#fff; }

/* Inputs, forms, expanders */
[data-testid="stForm"], [data-testid="stExpander"] details, [data-testid="stFileUploaderDropzone"] {
  border-radius:16px !important; border:1px solid #E9E5FB !important; background:#fff; }
[data-testid="stExpander"] summary:hover { color:#7C3AED; }
[data-testid="stMetric"] { background:#fff; border-radius:16px; padding:.8rem 1rem; border:1px solid #EDE9FE; }

/* Hero */
.hero { position:relative; overflow:hidden; border-radius:24px; padding:1.6rem 1.8rem; color:#fff; margin-bottom:1.1rem;
  background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 45%, #DB2777 100%); box-shadow: 0 18px 40px rgba(79,70,229,.25); }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.28), transparent 70%); }
.hero:before { content:""; position:absolute; left:40%; bottom:-90px; width:220px; height:220px; border-radius:50%;
  background: radial-gradient(circle, rgba(251,191,36,.35), transparent 70%); }
.hero h1 { color:#fff; font-size:2rem; font-weight:800; margin:0 0 .25rem; }
.hero p { margin:0; opacity:.92; font-size:1rem; max-width:720px; }
.hero-stats { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1rem; position:relative; z-index:1; }
.hero-stat { background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px);
  border-radius:14px; padding:.45rem .85rem; font-size:.85rem; }
.hero-stat b { font-size:1.05rem; margin-right:.25rem; }

/* Pipeline stepper */
.pipe { display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin: 0 0 1.1rem; }
.step { display:flex; align-items:center; gap:.45rem; padding:.4rem .75rem .4rem .45rem; border-radius:999px;
  background:#fff; border:1px solid #E9E5FB; font-size:.82rem; font-weight:600; color:#6B7280; }
.step .dot { width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-size:.75rem;
  background:#EEF2FF; color:#6366F1; }
.step.done { color:#1E1B4B; border-color:transparent; box-shadow: 0 4px 12px rgba(30,27,75,.08); }
.step.done .dot { color:#fff; }
.arrow { color:#C4B5FD; font-weight:700; }

/* Cards */
.card { background:#fff; border-radius:18px; padding:1.1rem 1.2rem; border:1px solid #EDE9FE;
  box-shadow: 0 6px 20px rgba(30,27,75,.06); margin-bottom:.8rem; }
.card.accent { border-left:6px solid var(--c, #8B5CF6); }
.sec { display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; margin:1.2rem 0 .7rem; }
.sec h3 { white-space:nowrap; }
@media (max-width: 640px) {
  .sec span.sub { flex-basis:100%; margin-left:0; }
  .bars { grid-template-columns: 1fr; }
  .hero { padding:1.2rem 1.1rem; } .hero h1 { font-size:1.5rem; }
  .job { gap:.7rem; } .ring { width:56px; height:56px; } .ring span { width:44px; height:44px; font-size:.9rem; }
}
.sec .ic { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; font-size:1.15rem; color:#fff;
  background: var(--c); box-shadow: 0 6px 14px rgba(0,0,0,.12); }
.sec h3 { margin:0; font-size:1.3rem; font-weight:800; }
.sec span.sub { color:#6B7280; font-size:.86rem; margin-left:.2rem; }

/* Stat tiles */
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:.75rem; margin:.4rem 0 1rem; }
.tile { border-radius:18px; padding:.9rem 1rem; color:#fff; position:relative; overflow:hidden; box-shadow: 0 10px 22px rgba(30,27,75,.12); }
.tile .v { font-size:1.7rem; font-weight:800; line-height:1.1; }
.tile .l { font-size:.8rem; opacity:.92; font-weight:600; }
.tile .i { position:absolute; right:.7rem; top:.55rem; font-size:1.5rem; opacity:.55; }

/* Chips */
.chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:.35rem 0; }
.chip { padding:.22rem .62rem; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid transparent; }
.chip.ok { background:#D1FAE5; color:#047857; border-color:#A7F3D0; }
.chip.rel { background:#FEF3C7; color:#B45309; border-color:#FDE68A; }
.chip.miss { background:#FFE4E6; color:#BE123C; border-color:#FECDD3; }
.chip.indigo { background:#E0E7FF; color:#4338CA; } .chip.violet { background:#EDE9FE; color:#6D28D9; }
.chip.pink { background:#FCE7F3; color:#BE185D; } .chip.sky { background:#E0F2FE; color:#0369A1; }
.chip.emerald { background:#D1FAE5; color:#047857; } .chip.amber { background:#FEF3C7; color:#B45309; }
.legend { font-size:.75rem; color:#6B7280; font-weight:600; margin-right:.2rem; }

/* Profile card */
.prof { display:flex; gap:1rem; align-items:center; }
.avatar { width:64px; height:64px; border-radius:20px; display:grid; place-items:center; font-weight:800; font-size:1.4rem; color:#fff;
  background: linear-gradient(135deg, #6366F1, #EC4899); box-shadow: 0 8px 20px rgba(99,102,241,.35); flex-shrink:0; }
.prof h2 { margin:0; font-size:1.35rem; font-weight:800; }
.prof .hl { color:#6B7280; font-size:.92rem; }
.kv { font-size:.9rem; margin:.25rem 0; color:#374151; } .kv b { color:#1E1B4B; }

/* Job card */
.job { display:flex; gap:1rem; align-items:flex-start; }
.ring { --p:50; --c:#8B5CF6; width:70px; height:70px; border-radius:50%; flex-shrink:0; display:grid; place-items:center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), #EEF2FF 0); }
.ring span { width:54px; height:54px; border-radius:50%; background:#fff; display:grid; place-items:center;
  font-weight:800; font-size:1.05rem; color:#1E1B4B; }
.job .t { font-weight:800; font-size:1.05rem; color:#1E1B4B; margin:0; }
.job .co { color:#6B7280; font-size:.86rem; margin:.1rem 0 .3rem; }
.job .meta { display:flex; flex-wrap:wrap; gap:.4rem; margin:.15rem 0 .2rem; }
.tag { font-size:.74rem; font-weight:600; padding:.18rem .55rem; border-radius:8px; background:#F3F4F6; color:#374151; }
.tag.remote { background:#E0F2FE; color:#0369A1; } .tag.sal { background:#ECFDF5; color:#047857; }
.badge-top { font-size:.72rem; font-weight:800; color:#fff; padding:.2rem .55rem; border-radius:8px;
  background: linear-gradient(135deg, #F59E0B, #EC4899); margin-left:.4rem; vertical-align:middle; }
.why { font-size:.85rem; color:#4B5563; margin-top:.35rem; }
.bars { display:grid; grid-template-columns: repeat(3, 1fr); gap:.6rem; margin-top:.5rem; }
.bar .lb { font-size:.72rem; color:#6B7280; font-weight:600; display:flex; justify-content:space-between; }
.bar .tr { height:7px; background:#EEF2FF; border-radius:99px; overflow:hidden; }
.bar .fl { height:100%; border-radius:99px; }

/* Decision banner */
.decision { border-radius:18px; padding:1rem 1.2rem; color:#fff; margin:.4rem 0 1rem; display:flex; gap:.9rem; align-items:center;
  box-shadow: 0 12px 26px rgba(30,27,75,.15); }
.decision .e { font-size:1.8rem; }
.decision b { font-size:1.05rem; }
.decision.recommend { background: linear-gradient(120deg, #059669, #10B981 60%, #34D399); }
.decision.prepare { background: linear-gradient(120deg, #4F46E5, #7C3AED 60%, #DB2777); }
.decision.search_more { background: linear-gradient(120deg, #D97706, #F59E0B 60%, #FB923C); }

/* Numbered tips & questions */
.tip { display:flex; gap:.7rem; align-items:flex-start; background:#fff; border:1px solid #EDE9FE; border-radius:14px;
  padding:.65rem .85rem; margin-bottom:.5rem; }
.tip .n { width:26px; height:26px; flex-shrink:0; border-radius:8px; display:grid; place-items:center; color:#fff; font-weight:800; font-size:.8rem; }
.qcat { font-size:.7rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; padding:.18rem .5rem; border-radius:6px; color:#fff; }
.cvdoc { background:#fff; border-radius:18px; padding:1.2rem 1.5rem; border:1px solid #EDE9FE; border-top:6px solid #8B5CF6; }
.empty { text-align:center; padding:2.2rem 1rem; border-radius:20px; border:2px dashed #DDD6FE; background: rgba(255,255,255,.6); color:#6B7280; }
.empty .e { font-size:2.2rem; } .empty b { color:#1E1B4B; display:block; margin:.3rem 0 .15rem; font-size:1.05rem; }
</style>
"""

CAT_COLORS = {"technical": INDIGO, "behavioural": PINK, "behavioral": PINK, "gap": AMBER, "motivation": EMERALD}
STEP_COLORS = [INDIGO, VIOLET, SKY, EMERALD, AMBER, PINK, ROSE]


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    render(CSS)


def score_color(score: float) -> str:
    return EMERALD if score >= 70 else (VIOLET if score >= 55 else (AMBER if score >= 40 else ROSE))


def hero(title: str, subtitle: str, stats: Sequence[Tuple[str, str]] = ()) -> None:
    s = "".join(f'<div class="hero-stat"><b>{esc(v)}</b>{esc(l)}</div>' for v, l in stats)
    render(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>'
           f'<div class="hero-stats">{s}</div></div>')


def pipeline(steps: Sequence[Tuple[str, str, bool]]) -> None:
    """steps: (emoji, label, done)."""
    parts = []
    for i, (emo, label, done) in enumerate(steps):
        c = STEP_COLORS[i % len(STEP_COLORS)]
        dot_style = f' style="background:{c}"' if done else ""
        parts.append(f'<div class="step{" done" if done else ""}"><span class="dot"{dot_style}>'
                     f'{"✓" if done else esc(emo)}</span>{esc(label)}</div>')
        if i < len(steps) - 1:
            parts.append('<span class="arrow">›</span>')
    render(f'<div class="pipe">{"".join(parts)}</div>')


def section(icon: str, title: str, color: str = VIOLET, sub: str = "") -> None:
    render(f'<div class="sec"><div class="ic" style="--c:{color}">{esc(icon)}</div>'
           f'<h3>{esc(title)}</h3>{f"<span class=sub>{esc(sub)}</span>" if sub else ""}</div>')


def tiles(items: Sequence[Tuple[str, str, str, str]]) -> None:
    """items: (value, label, icon, gradient-start color)."""
    grads = {INDIGO: "#818CF8", VIOLET: "#C084FC", PINK: "#F9A8D4", AMBER: "#FCD34D",
             EMERALD: "#6EE7B7", SKY: "#7DD3FC", ROSE: "#FDA4AF"}
    out = "".join(
        f'<div class="tile" style="background:linear-gradient(135deg,{c},{grads.get(c, c)})">'
        f'<div class="i">{esc(i)}</div><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div></div>'
        for v, l, i, c in items)
    render(f'<div class="tiles">{out}</div>')


def chips_html(items: Iterable[str], kind: Optional[str] = None) -> str:
    items = list(items)
    if not items:
        return '<span class="chip" style="background:#F3F4F6;color:#9CA3AF">none</span>'
    return "".join(f'<span class="chip {kind or CHIP_CYCLE[i % len(CHIP_CYCLE)]}">{esc(x)}</span>'
                   for i, x in enumerate(items))


def chips(items: Iterable[str], kind: Optional[str] = None, label: str = "") -> None:
    lab = f'<span class="legend">{esc(label)}</span>' if label else ""
    render(f'<div class="chips">{lab}{chips_html(items, kind)}</div>')


def empty_state(emoji: str, title: str, text: str) -> None:
    render(f'<div class="empty"><div class="e">{esc(emoji)}</div><b>{esc(title)}</b>{esc(text)}</div>')


def profile_card(p) -> None:
    initials = "".join(w[0] for w in (p.name or "You").split()[:2]).upper()
    locs = ", ".join(p.preferred_locations) or "Anywhere"
    render(
        '<div class="card accent" style="--c:#8B5CF6">'
        f'<div class="prof"><div class="avatar">{esc(initials)}</div><div>'
        f'<h2>{esc(p.name or "Your career profile")}</h2><div class="hl">{esc(p.headline or "")}</div></div></div>'
        f'<div class="kv" style="margin-top:.8rem">{esc(p.summary)}</div>'
        f'<div class="kv"><b>🎯 Target roles:</b> {esc(", ".join(p.preferred_roles) or "—")}</div>'
        f'<div class="kv"><b>📍 Locations:</b> {esc(locs)}{" · 🌐 open to remote" if p.remote_ok else ""}</div>'
        f'<div class="kv"><b>🚀 Goal:</b> {esc(p.career_goal or "—")}</div>'
        f'<div class="chips" style="margin-top:.6rem">{chips_html(p.skills)}</div></div>')


def _bar(label: str, value: float, color: str) -> str:
    return (f'<div class="bar"><div class="lb"><span>{esc(label)}</span><span>{value:.0f}</span></div>'
            f'<div class="tr"><div class="fl" style="width:{max(0, min(100, value)):.0f}%;background:{color}"></div></div></div>')


def job_card(job, match, recommended: bool, rank: int) -> None:
    c = score_color(match.score)
    meta = [f'<span class="tag">📍 {esc(job.location or "—")}</span>']
    if job.remote:
        meta.append('<span class="tag remote">🌐 Remote</span>')
    if job.salary:
        meta.append(f'<span class="tag sal">💰 {esc(job.salary)}</span>')
    top = '<span class="badge-top">★ TOP PICK</span>' if recommended else ""
    render(
        f'<div class="card accent" style="--c:{c}"><div class="job">'
        f'<div class="ring" style="--p:{match.score:.0f};--c:{c}"><span>{match.score:.0f}</span></div>'
        f'<div style="flex:1;min-width:0"><p class="t">#{rank} {esc(job.title)}{top}</p>'
        f'<div class="co">🏢 {esc(job.company or "Company not listed")}</div>'
        f'<div class="meta">{"".join(meta)}</div>'
        f'<div class="chips"><span class="legend">Matched</span>{chips_html(match.matched_skills, "ok")}</div>'
        + (f'<div class="chips"><span class="legend">Related</span>{chips_html(match.related_skills, "rel")}</div>'
           if match.related_skills else "")
        + f'<div class="chips"><span class="legend">Missing</span>{chips_html(match.missing_skills[:8], "miss")}</div>'
        f'<div class="bars">{_bar("Skills", match.skill_score, INDIGO)}'
        f'{_bar("Semantic fit", match.semantic_score, PINK)}{_bar("Experience", match.experience_score, EMERALD)}</div>'
        f'<div class="why">💡 {esc(match.explanation)}</div></div></div></div>')


def decision_banner(decision) -> None:
    emoji = {"recommend": "✅", "prepare": "🧭", "search_more": "🔁"}.get(decision.action, "🤖")
    title = decision.action.replace("_", " ").title()
    render(f'<div class="decision {esc(decision.action)}"><div class="e">{emoji}</div>'
           f'<div><b>Decision Agent · {esc(title)}</b><br>{esc(decision.reason)}</div></div>')


def numbered(items: Sequence[str]) -> None:
    colors = [INDIGO, VIOLET, PINK, SKY, EMERALD, AMBER]
    render("".join(f'<div class="tip"><div class="n" style="background:{colors[i % len(colors)]}">{i + 1}</div>'
                   f'<div>{esc(t)}</div></div>' for i, t in enumerate(items)))


def question_header(i: int, category: str, question: str) -> None:
    c = CAT_COLORS.get(category.lower(), VIOLET)
    render(f'<div style="display:flex;gap:.6rem;align-items:center;margin:.2rem 0">'
           f'<span class="qcat" style="background:{c}">{esc(category)}</span>'
           f'<b style="color:#1E1B4B">Q{i}. {esc(question)}</b></div>')


def sidebar_brand(llm_on: bool, mode: str) -> None:
    render('<div class="sb-brand"><div class="sb-logo">🎯</div><div><div class="sb-title">Intelligent<br>Job Search</div>'
           '<div class="sb-sub">Multi-agent career assistant</div></div></div>'
           f'<div class="sb-pill {"on" if llm_on else "off"}">{"🟢" if llm_on else "🟠"} {esc(mode)}</div>')


def sidebar_flow() -> None:
    render('<div class="sb-flow"><b>How it works</b><br>👤 Profile → 🔍 Search → 📄 Analysis → 🎯 Matching → '
           '🧠 Decision → ✍️ CV &amp; 🎤 Interview → 📊 Dashboard → 💾 Memory</div>')
