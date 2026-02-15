# app.py
# ------------------------------------------------------------
# ✅ Features
# - RTL Persian UI + stable sliders (force slider LTR)
# - No sidebar (fixes right collapsible panel issues)
# - Advanced filters (auto-apply, no Apply button) + ITM/ATM/OTM
# - Column picker + column ORDER for chain table + selected contract details
# - Chain table includes: BSM price (no decimals) + % gap to market + % change vs yesterday
# - Underlying top metrics include % change vs yesterday (colored via delta)
# - Extra sliders for key numeric/percent fields (bid/ask, %chg, bsm gap)
# - BSM + IV + Greeks (tab 2)
# - History tab: REAL EOD price history for option + underlying (finpy-tse) + tables + charts
#
# Install:
#   pip install streamlit plotly pandas tseopt lxml jdatetime
# History:
#   pip install finpy-tse
# Run:
#   streamlit run app.py
# ------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from tseopt import get_all_options_data, fetch_historical_lob
from tseopt.use_case.options_chains import Chains
import jdatetime
import re


# -----------------------------
# Page + RTL + slider fix
# -----------------------------
st.set_page_config(page_title="آپشن‌ها", layout="wide")

st.markdown(
    """
<style>
/* RTL for app */TAB 1: Advanced filters
html, body, [class*="css"] { direction: rtl !important; text-align: right !important; }
label, p, div, span, input, textarea { direction: rtl !important; text-align: right !important; }
* { font-variant-numeric: tabular-nums; }
/* جلوگیری از برعکس شدن علامت منفی */
.num-ltr { 
    direction: ltr !important; 
    unicode-bidi: embed !important; 
    display: inline-block; 
}

/* --- Slider Fix (Windows + Chrome + RTL) ---
   Force slider container LTR strongly, while keeping text RTL.
*/
div[data-testid="stSlider"] { direction: ltr !important; unicode-bidi: embed !important; }
div[data-testid="stSlider"] * { direction: ltr !important; unicode-bidi: embed !important; }

/* BaseWeb slider internals */
[data-baseweb="slider"] { direction: ltr !important; unicode-bidi: embed !important; }
[data-baseweb="slider"] * { direction: ltr !important; unicode-bidi: embed !important; }

/* Keep slider label RTL */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label *,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span {
  direction: rtl !important;
  text-align: right !important;
  unicode-bidi: plaintext !important;
}

.small { opacity: 0.75; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


st.title("📈 زنجیره اختیار معامله")

with st.expander("🔄 تنظیمات بروزرسانی خودکار", expanded=False):
    c1, c2, c3 = st.columns([1.2, 1, 2.8])

    with c1:
        auto_on = st.toggle(
            "فعال‌سازی بروزرسانی خودکار",
            value=st.session_state.get("auto_refresh_on", True),
            key="auto_refresh_on",
        )

    with c2:
        interval_sec = st.number_input(
            "هر چند ثانیه؟",
            min_value=5,
            max_value=600,
            value=int(st.session_state.get("auto_refresh_sec", 60)),
            step=5,
            key="auto_refresh_sec",
        )

    with c3:
        st.caption("اگر فعال باشد، صفحه هر X ثانیه یک‌بار رفرش می‌شود و داده‌ها با TTL کش‌ها بروز می‌شوند.")

# بیرون expander هم می‌تونی وضعیت رو خلاصه نشون بدی (اختیاری)
st.caption(f"وضعیت بروزرسانی: {'فعال' if st.session_state.get('auto_refresh_on', True) else 'خاموش'} | "
           f"هر {int(st.session_state.get('auto_refresh_sec', 60))} ثانیه")

if st.session_state.get("auto_refresh_on", True):
    st_autorefresh(interval=int(st.session_state.get("auto_refresh_sec", 60)) * 1000, key="auto_refresh_counter")

# -----------------------------
# Helpers
# -----------------------------
_num_like_re = re.compile(r"""^\s*[-−]?\s*[\d,]+(\.\d+)?\s*(%|٪)?\s*$""")

def wrap_num_ltr(s):
    """Wrap numeric-like strings to prevent minus sign flipping in RTL tables."""
    if s is None or _is_na(s):
        return "—"
    txt = str(s).strip()
    if txt == "—" or txt == "":
        return txt
    if _num_like_re.match(txt):
        # normalize Arabic percent if any - optional
        return f"<span class='num-ltr'>{txt}</span>"
    return txt

def _is_na(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None


def to_int(x):
    if _is_na(x):
        return None
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        xf = float(x)
        if np.isfinite(xf) and abs(xf - round(xf)) < 1e-9:
            return int(round(xf))
        return int(x)
    except Exception:
        return None


def to_float(x):
    if _is_na(x):
        return None
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except Exception:
        return None


def fmt_num(x, decimals=2):
    if x is None or _is_na(x):
        return "—"
    try:
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        xf = float(x)
        if not np.isfinite(xf):
            return "—"
        if abs(xf - round(xf)) < 1e-9:
            return f"{int(round(xf)):,}"
        return f"{xf:,.{decimals}f}"
    except Exception:
        return str(x)


def safe_str(x):
    if x is None or _is_na(x):
        return "—"
    return str(x)


def fmt_date_yyyymmdd(x):
    if x is None or _is_na(x):
        return "—"
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}/{s[4:6]}/{s[6:]}"
    return s


def gregorian_yyyymmdd_to_jalali_str(x):
    if x is None or _is_na(x):
        return "—"
    s = str(x).strip()
    if len(s) != 8 or not s.isdigit():
        return "—"
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:])
    try:
        jd = jdatetime.date.fromgregorian(date=pd.Timestamp(y, m, d).date())
        return f"{jd.year:04d}/{jd.month:02d}/{jd.day:02d}"
    except Exception:
        return "—"


def map_option_type(v: str) -> str:
    v = (v or "").strip().lower()
    if v == "call":
        return "اختیار خرید"
    if v == "put":
        return "اختیار فروش"
    return "—"


def sanitize_multiselect_state(key: str, options: list, default: list):
    prev = st.session_state.get(key, None)
    if prev is None:
        st.session_state[key] = [x for x in default if x in options]
    else:
        st.session_state[key] = [x for x in prev if x in options]


def safe_range_slider_int(label, series: pd.Series, key: str):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        st.caption(f"{label}: موجود نیست")
        return None
    mn, mx = int(s.min()), int(s.max())
    if mn == mx:
        v = st.number_input(label, min_value=mn, max_value=mx, value=mn, step=1, disabled=True, key=key + "_single")
        return (v, v)
    return st.slider(label, mn, mx, (mn, mx), key=key)


def safe_range_slider_float(label, series: pd.Series, key: str, decimals=2):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        st.caption(f"{label}: موجود نیست")
        return None
    mn, mx = float(s.min()), float(s.max())
    if mn == mx:
        v = st.number_input(label, value=mn, disabled=True, key=key + "_single")
        return (v, v)
    step = 10 ** (-decimals)
    return st.slider(label, mn, mx, (mn, mx), step=step, key=key)


# -----------------------------
# HTML table renderer (no truncation)
# -----------------------------
def render_table_html(df: pd.DataFrame, height_px: int = 560, title: str | None = None):
    if title:
        st.markdown(f"### {title}")
    if df is None or len(df) == 0:
        st.info("داده‌ای برای نمایش وجود ندارد.")
        return

    html_table = df.to_html(index=False, escape=False)
    html = f"""
<!doctype html><html><head><meta charset="utf-8">
<style>
  body {{ margin:0; direction:rtl; text-align:right; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
  .num-ltr {{ direction:ltr !important; unicode-bidi:embed !important; display:inline-block; }}
  .wrap {{ border:1px solid rgba(0,0,0,0.12); border-radius:12px; overflow:auto; max-height:{height_px}px; }}
  table {{ border-collapse:collapse; width:max-content; min-width:100%; table-layout:auto; font-size:14px; }}
  thead th {{ position:sticky; top:0; background:rgba(255,255,255,0.98); z-index:2; border-bottom:1px solid rgba(0,0,0,0.18); }}
  th, td {{ padding:8px 10px; border-bottom:1px solid rgba(0,0,0,0.08); white-space:nowrap; text-align:right; font-variant-numeric:tabular-nums; }}
  tr:hover td {{ background:rgba(0,0,0,0.03); }}
</style></head><body><div class="wrap">{html_table}</div></body></html>
"""

    components.html(html, height=height_px + 40, scrolling=True)


# -----------------------------
# BSM + IV (no scipy)
# -----------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bsm_price_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    if any(v is None for v in [S, K, T, r, sigma]) or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        price = S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = math.exp(-q * T) * _norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
        delta = -math.exp(-q * T) * _norm_cdf(-d1)
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)

    gamma = math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)
    theta = -(S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))

    return {"قیمت نظری (بلک-شولز)": price, "دلتا": delta, "گاما": gamma, "وگا": vega, "تتا": theta, "رو": rho}


def implied_vol_bisection(market_price, S, K, T, r, option_type="call", q=0.0, lo=1e-6, hi=5.0, iters=70):
    if any(v is None for v in [market_price, S, K, T, r]) or market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    def price_at(sig):
        res = bsm_price_greeks(S, K, T, r, sig, option_type=option_type, q=q)
        return None if res is None else res["قیمت نظری (بلک-شولز)"]

    p_lo, p_hi = price_at(lo), price_at(hi)
    if p_lo is None or p_hi is None:
        return None
    if not (p_lo <= market_price <= p_hi):
        return None

    a, b = lo, hi
    for _ in range(iters):
        mid = (a + b) / 2.0
        p_mid = price_at(mid)
        if p_mid is None:
            return None
        if abs(p_mid - market_price) < 1e-6:
            return mid
        if p_mid < market_price:
            a = mid
        else:
            b = mid
    return (a + b) / 2.0


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(ttl=5)
def load_entire_market():
    df = get_all_options_data()

    for c in ["ua_tse_code", "tse_code", "ticker", "ua_ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    num_cols = [
        "days_to_maturity", "strike_price",
        "last_price", "close_price", "yesterday_price",
        "bid_price", "bid_volume", "ask_price", "ask_volume",
        "open_positions", "contract_size", "notional_value",
        "ua_last_price", "ua_close_price", "ua_yesterday_price",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(ttl=60)
def build_underlyings(entire_df: pd.DataFrame) -> pd.DataFrame:
    chains = Chains(entire_df)
    ua = chains.underlying_asset_info.copy()
    for c in ["ua_tse_code", "ua_ticker"]:
        if c in ua.columns:
            ua[c] = ua[c].astype(str)

    if "ua_name" in ua.columns:
        ua["label"] = ua["ua_ticker"].astype(str) + " — " + ua["ua_name"].astype(str)
    else:
        ua["label"] = ua["ua_ticker"].astype(str)
    return ua.sort_values("label")


# -----------------------------
# Top controls (NO SIDEBAR)
# -----------------------------
entire = load_entire_market()
ua_info = build_underlyings(entire)

top1, top2, top3 = st.columns([2, 1, 1])
with top1:
    ua_label = st.selectbox("دارایی پایه", ua_info["label"].tolist(), key="top_ua")
with top2:
    option_type_raw = st.selectbox("نوع اختیار", ["both", "call", "put"], index=0, key="top_type")
with top3:
    if st.button("🔄 بروزرسانی داده‌ها"):
        st.cache_data.clear()
        st.rerun()

ua_row = ua_info.loc[ua_info["label"] == ua_label].iloc[0]
ua_tse_code = safe_str(ua_row.get("ua_tse_code"))
ua_ticker = safe_str(ua_row.get("ua_ticker"))
ua_name = safe_str(ua_row.get("ua_name")) if "ua_name" in ua_row.index else "—"

chains = Chains(entire)
options_df = chains.options(ua_tse_code=ua_tse_code, option_type=option_type_raw).copy()
if options_df is None or len(options_df) == 0:
    st.warning("برای این دارایی پایه، آپشنی پیدا نشد.")
    st.stop()

for c in ["ticker", "tse_code", "ua_ticker", "ua_tse_code"]:
    if c in options_df.columns:
        options_df[c] = options_df[c].astype(str)

options_df["نوع اختیار"] = options_df["option_type"].apply(map_option_type) if "option_type" in options_df.columns else "—"
options_df["سررسید (میلادی)"] = options_df["end_date"].apply(fmt_date_yyyymmdd) if "end_date" in options_df.columns else "—"
options_df["سررسید (شمسی)"] = options_df["end_date"].apply(gregorian_yyyymmdd_to_jalali_str) if "end_date" in options_df.columns else "—"

ua_last = None
try:
    ua_last = float(entire.loc[entire["ua_tse_code"] == ua_tse_code, "ua_last_price"].iloc[0])
except Exception:
    ua_last = None


# -----------------------------
# Tabs
# -----------------------------
tab_filters, tab_bsm, tab_strat, tab_history = st.tabs(["📌 فیلترهای پیشرفته", "🧮 بلک-شولز + IV + Greeks", "🎯 استراتژی‌ها", "📉 تاریخچه قرارداد"])


# ============================================================
# TAB 1: Advanced filters  (FULL / CLEAN / NO SCOPE ISSUES)
# ============================================================
with tab_filters:
    with st.expander("ℹ️ اطلاعات دارایی پایه", expanded=True):
        st.subheader(f"دارایی پایه: {ua_ticker} | {ua_name}")

        # -----------------------------
        # Top metrics (Underlying)
        # -----------------------------
        a, b, c, d = st.columns(4)

        try:
            v_last = float(entire.loc[entire["ua_tse_code"] == ua_tse_code, "ua_last_price"].iloc[0])
            a.metric("قیمت آخرین دارایی پایه", fmt_num(to_int(v_last)))
        except Exception:
            v_last = None
            a.metric("قیمت آخرین دارایی پایه", "—")

        try:
            v_close = float(entire.loc[entire["ua_tse_code"] == ua_tse_code, "ua_close_price"].iloc[0])
            b.metric("قیمت پایانی دارایی پایه", fmt_num(to_int(v_close)))
        except Exception:
            v_close = None
            b.metric("قیمت پایانی دارایی پایه", "—")

        try:
            v_yest = float(entire.loc[entire["ua_tse_code"] == ua_tse_code, "ua_yesterday_price"].iloc[0])
            c.metric("قیمت دیروز دارایی پایه", fmt_num(to_int(v_yest)))
        except Exception:
            v_yest = None
            c.metric("قیمت دیروز دارایی پایه", "—")

        # ✅ colored % change (RTL safe)
        try:
            if v_last is not None and v_yest is not None and float(v_yest) != 0:
                chg = (float(v_last) - float(v_yest)) / float(v_yest) * 100.0

                if chg > 0:
                    bg, fg = "#e8f5e9", "#1b5e20"   # green
                elif chg < 0:
                    bg, fg = "#ffebee", "#b71c1c"   # red
                else:
                    bg, fg = "#eeeeee", "#424242"   # gray

                d.markdown(
                    f"""
                    <div style="border:1px solid rgba(0,0,0,0.12);border-radius:12px;padding:10px 12px;">
                    <div style="font-size:0.95rem;opacity:.85;margin-bottom:6px;">درصد تغییر نسبت به دیروز</div>
                    <div class="num-ltr" style="display:inline-block;padding:6px 12px;border-radius:999px;background:{bg};color:{fg};
                                font-weight:900;font-variant-numeric:tabular-nums;">
                        {chg:,.2f}%
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                d.metric("درصد تغییر نسبت به دیروز", "—")
        except Exception:
            d.metric("درصد تغییر نسبت به دیروز", "—")

    # -----------------------------
    # Inputs UI (expanders)
    # -----------------------------
    # place-holders for percent sliders (need computed cols)
    pct_slider_ph = None
    gap_slider_ph = None

    with st.expander("🔎 فیلترهای اصلی", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            q_key = "flt_q"
            q = st.text_input("جستجو در نماد آپشن", value=st.session_state.get(q_key, ""), key=q_key).strip()

        with f2:
            t_key = "flt_type"
            sanitize_multiselect_state(t_key, ["اختیار خرید", "اختیار فروش"], ["اختیار خرید", "اختیار فروش"])
            typ_selected = st.multiselect("نوع اختیار", ["اختیار خرید", "اختیار فروش"], key=t_key)

        with f3:
            mats = sorted([m for m in options_df["سررسید (شمسی)"].dropna().unique().tolist() if m != "—"])
            m_key = "flt_mats"
            sanitize_multiselect_state(m_key, mats, st.session_state.get(m_key, []))
            chosen_mats = st.multiselect("سررسید (شمسی)", mats, key=m_key)

        with f4:
            liq_key = "flt_liq"
            only_liquid = st.checkbox("فقط دارای bid یا ask", value=st.session_state.get(liq_key, False), key=liq_key)

    with st.expander("📊 وضعیت قرارداد (ITM / ATM / OTM)", expanded=False):
        it1, it2 = st.columns([2, 1])
        with it1:
            ms_key = "flt_mstatus"
            sanitize_multiselect_state(ms_key, ["ITM", "ATM", "OTM"], ["ITM", "ATM", "OTM"])
            m_status = st.multiselect("فیلتر وضعیت", ["ITM", "ATM", "OTM"], key=ms_key)
        with it2:
            thr_key = "flt_atmthr"
            atm_thr = st.number_input("آستانه ATM (%)", 0.0, 10.0, float(st.session_state.get(thr_key, 1.0)), 0.1, key=thr_key)

    with st.expander("📏 فیلترهای بازه‌ای + مرتب‌سازی", expanded=False):
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            dtm_range = safe_range_slider_int("بازه مانده تا سررسید (روز)", options_df.get("days_to_maturity", pd.Series([])), key="flt_dtm")
        with r2:
            k_range = safe_range_slider_int("بازه قیمت اعمال", options_df.get("strike_price", pd.Series([])), key="flt_k")
        with r3:
            p_range = safe_range_slider_int("بازه قیمت پریمیوم (آخرین)", options_df.get("last_price", pd.Series([])), key="flt_p")
        with r4:
            oi_range = safe_range_slider_int("بازه موقعیت باز (OI)", options_df.get("open_positions", pd.Series([])), key="flt_oi")

        s1, s2 = st.columns(2)
        with s1:
            sort_by = st.selectbox(
                "مرتب‌سازی بر اساس",
                [
                    "مانده تا سررسید",
                    "قیمت اعمال",
                    "قیمت پریمیوم (آخرین)",
                    "موقعیت باز",
                    "درصد تغییر پریمیوم نسبت به دیروز",
                    "اختلاف درصدی بازار با بلک-شولز",
                ],
                index=0,
                key="flt_sort",
            )
        with s2:
            asc = st.checkbox("صعودی", value=st.session_state.get("flt_asc", True), key="flt_asc")

    with st.expander("🧮 تنظیمات بلک-شولز (جدول زنجیره)", expanded=False):
        bb1, bb2, bb3, bb4 = st.columns(4)
        with bb1:
            st.number_input("نرخ بهره r", 0.0, 2.0, float(st.session_state.get("bsm_r_chain", 0.30)), 0.01, key="bsm_r_chain")
        with bb2:
            st.number_input("سود نقدی q", 0.0, 2.0, float(st.session_state.get("bsm_q_chain", 0.00)), 0.01, key="bsm_q_chain")
        with bb3:
            st.number_input("نوسان σ", 0.0001, 5.0, float(st.session_state.get("bsm_sigma_chain", 0.60)), 0.05, key="bsm_sigma_chain")
        with bb4:
            basis_choice = st.selectbox(
                "مبنای سال",
                ["365", "252"],
                index=0 if float(st.session_state.get("bsm_basis_chain", 365.0)) == 365.0 else 1,
                key="bsm_basis_choice_chain",
            )
        st.session_state["bsm_basis_chain"] = 365.0 if basis_choice == "365" else 252.0

        st.caption(
            "قیمت بلک-شولز در جدول با S=آخرین دارایی پایه، قیمت اعمال=قیمت اعمال، T=مانده/مبنای سال "
            "و پارامترهای r، σ و q محاسبه می‌شود. (نمایش قیمت بلک-شولز بدون اعشار است.)"
        )

    with st.expander("🔎 فیلترهای تکمیلی (bid/ask و درصدها)", expanded=False):
        x1, x2, x3, x4 = st.columns(4)
        with x1:
            bid_range = safe_range_slider_int("بازه بهترین قیمت خرید", options_df.get("bid_price", pd.Series([])), key="flt_bid")
        with x2:
            ask_range = safe_range_slider_int("بازه بهترین قیمت فروش", options_df.get("ask_price", pd.Series([])), key="flt_ask")

        # placeholders (will render after computing pct + bsm gap)
        with x3:
            pct_slider_ph = st.empty()
        with x4:
            gap_slider_ph = st.empty()

    # -----------------------------
    # Apply filters (ONE PLACE) — no button
    # -----------------------------
    filtered = options_df.copy()

    # basic filters
    if q:
        filtered = filtered[filtered["ticker"].astype(str).str.contains(q, case=False, na=False)]
    if typ_selected:
        filtered = filtered[filtered["نوع اختیار"].isin(typ_selected)]
    if chosen_mats:
        filtered = filtered[filtered["سررسید (شمسی)"].isin(chosen_mats)]

    # range filters
    if dtm_range and "days_to_maturity" in filtered.columns:
        filtered = filtered[(filtered["days_to_maturity"] >= dtm_range[0]) & (filtered["days_to_maturity"] <= dtm_range[1])]
    if k_range and "strike_price" in filtered.columns:
        filtered = filtered[(filtered["strike_price"] >= k_range[0]) & (filtered["strike_price"] <= k_range[1])]
    if p_range and "last_price" in filtered.columns:
        filtered = filtered[(filtered["last_price"] >= p_range[0]) & (filtered["last_price"] <= p_range[1])]
    if oi_range and "open_positions" in filtered.columns:
        filtered = filtered[(filtered["open_positions"] >= oi_range[0]) & (filtered["open_positions"] <= oi_range[1])]

    if only_liquid and {"bid_price", "ask_price"}.issubset(set(filtered.columns)):
        filtered = filtered[((filtered["bid_price"].fillna(0) > 0) | (filtered["ask_price"].fillna(0) > 0))]

    # -----------------------------
    # % change vs yesterday (premium)
    # -----------------------------
    if {"last_price", "yesterday_price"}.issubset(set(filtered.columns)):
        y = pd.to_numeric(filtered["yesterday_price"], errors="coerce")
        l = pd.to_numeric(filtered["last_price"], errors="coerce")
        filtered["pct_change_vs_yesterday"] = np.where((y.notna()) & (y != 0) & (l.notna()), (l - y) / y * 100.0, np.nan)
    else:
        filtered["pct_change_vs_yesterday"] = np.nan

    # -----------------------------
    # ITM/ATM/OTM status
    # -----------------------------
    filtered["وضعیت"] = "نامشخص"
    if ua_last is not None and "strike_price" in filtered.columns:
        S_ua = float(ua_last)
        K_ser = pd.to_numeric(filtered["strike_price"], errors="coerce").replace(0, np.nan)

        if K_ser.notna().any():
            rel = (abs(S_ua - K_ser) / S_ua)
            is_atm = rel <= (float(atm_thr) / 100.0)

            is_call = filtered["نوع اختیار"].astype(str).str.contains("خرید", na=False)
            is_put = filtered["نوع اختیار"].astype(str).str.contains("فروش", na=False)

            itm_call = is_call & (S_ua > K_ser)
            itm_put = is_put & (S_ua < K_ser)
            otm_call = is_call & (S_ua < K_ser)
            otm_put = is_put & (S_ua > K_ser)

            is_itm = (itm_call | itm_put) & (~is_atm)
            is_otm = (otm_call | otm_put) & (~is_atm)

            status = pd.Series("نامشخص", index=filtered.index, dtype="object")
            status[is_itm] = "ITM"
            status[is_atm] = "ATM"
            status[is_otm] = "OTM"
            filtered["وضعیت"] = status

    if m_status:
        keep = set(m_status) | {"نامشخص"}
        filtered = filtered[filtered["وضعیت"].isin(keep)]

    # -----------------------------
    # BSM price + gap% (chain)
    # -----------------------------
    BSM_R = float(st.session_state.get("bsm_r_chain", 0.30))
    BSM_Q = float(st.session_state.get("bsm_q_chain", 0.00))
    BSM_SIGMA = float(st.session_state.get("bsm_sigma_chain", 0.60))
    BSM_BASIS = float(st.session_state.get("bsm_basis_chain", 365.0))

    filtered["bsm_price"] = np.nan
    filtered["bsm_gap_pct"] = np.nan

    if ua_last is not None and {"strike_price", "days_to_maturity", "last_price"}.issubset(set(filtered.columns)):
        S = float(ua_last)
        K_series = pd.to_numeric(filtered["strike_price"], errors="coerce")
        days_series = pd.to_numeric(filtered["days_to_maturity"], errors="coerce")
        mp_series = pd.to_numeric(filtered["last_price"], errors="coerce")
        T_series = days_series / float(BSM_BASIS)
        is_call_series = filtered["نوع اختیار"].astype(str).str.contains("خرید", na=False)

        bsm_vals = []
        for k, t, callflag in zip(K_series.tolist(), T_series.tolist(), is_call_series.tolist()):
            if _is_na(k) or _is_na(t) or (k is None) or (t is None) or (k <= 0) or (t <= 0):
                bsm_vals.append(np.nan)
                continue
            res = bsm_price_greeks(
                S, float(k), float(t),
                float(BSM_R), float(BSM_SIGMA),
                option_type=("call" if callflag else "put"),
                q=float(BSM_Q),
            )
            bsm_vals.append(res["قیمت نظری (بلک-شولز)"] if res else np.nan)

        filtered["bsm_price"] = bsm_vals
        bp = pd.to_numeric(filtered["bsm_price"], errors="coerce")
        filtered["bsm_gap_pct"] = np.where(
            (bp.notna()) & (bp != 0) & (mp_series.notna()),
            (mp_series - bp) / bp * 100.0,
            np.nan,
        )

    # -----------------------------
    # Render percent sliders now (need computed cols)
    # -----------------------------
    chg_range = None
    gap_range = None
    if pct_slider_ph is not None:
        chg_range = pct_slider_ph.slider(
            "بازه درصد تغییر پریمیوم نسبت به دیروز",
            *(
                (float(pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().min()),
                 float(pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().max()))
                if pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().shape[0] > 0
                else (0.0, 0.0)
            ),
            value=(
                (float(pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().min()),
                 float(pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().max()))
                if pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().shape[0] > 0
                else (0.0, 0.0)
            ),
            key="flt_chg",
        ) if pd.to_numeric(filtered["pct_change_vs_yesterday"], errors="coerce").dropna().shape[0] > 0 else None

    if gap_slider_ph is not None:
        gap_range = gap_slider_ph.slider(
            "بازه اختلاف درصدی بازار با بلک-شولز",
            *(
                (float(pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().min()),
                 float(pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().max()))
                if pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().shape[0] > 0
                else (0.0, 0.0)
            ),
            value=(
                (float(pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().min()),
                 float(pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().max()))
                if pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().shape[0] > 0
                else (0.0, 0.0)
            ),
            key="flt_gap",
        ) if pd.to_numeric(filtered["bsm_gap_pct"], errors="coerce").dropna().shape[0] > 0 else None

    # apply extra filters
    if bid_range and "bid_price" in filtered.columns:
        filtered = filtered[(filtered["bid_price"] >= bid_range[0]) & (filtered["bid_price"] <= bid_range[1])]
    if ask_range and "ask_price" in filtered.columns:
        filtered = filtered[(filtered["ask_price"] >= ask_range[0]) & (filtered["ask_price"] <= ask_range[1])]
    if chg_range and "pct_change_vs_yesterday" in filtered.columns:
        filtered = filtered[(filtered["pct_change_vs_yesterday"] >= chg_range[0]) & (filtered["pct_change_vs_yesterday"] <= chg_range[1])]
    if gap_range and "bsm_gap_pct" in filtered.columns:
        filtered = filtered[(filtered["bsm_gap_pct"] >= gap_range[0]) & (filtered["bsm_gap_pct"] <= gap_range[1])]

    # -----------------------------
    # Sorting (final)
    # -----------------------------
    sort_map = {
        "مانده تا سررسید": "days_to_maturity",
        "قیمت اعمال": "strike_price",
        "قیمت پریمیوم (آخرین)": "last_price",
        "موقعیت باز": "open_positions",
        "درصد تغییر پریمیوم نسبت به دیروز": "pct_change_vs_yesterday",
        "اختلاف درصدی بازار با بلک-شولز": "bsm_gap_pct",
    }
    sc = sort_map.get(sort_by)
    if sc in filtered.columns:
        filtered = filtered.sort_values(sc, ascending=asc)

    # -----------------------------
    # Column picker + ORDER (GLOBAL)
    # -----------------------------
    with st.expander("🧩 تنظیم ستون‌های جدول (انتخاب و ترتیب)", expanded=False):
        col_map = {
            "ticker": "نماد آپشن",
            "tse_code": "کد TSE",
            "ua_ticker": "نماد دارایی پایه",
            "ua_tse_code": "کد TSE دارایی پایه",

            "نوع اختیار": "نوع اختیار",
            "سررسید (شمسی)": "سررسید (شمسی)",
            "سررسید (میلادی)": "سررسید (میلادی)",

            "days_to_maturity": "مانده تا سررسید (روز)",
            "strike_price": "قیمت اعمال",

            "last_price": "قیمت پریمیوم (آخرین)",
            "close_price": "قیمت پریمیوم (پایانی)",
            "yesterday_price": "قیمت پریمیوم (دیروز)",
            "pct_change_vs_yesterday": "درصد تغییر پریمیوم نسبت به دیروز",

            "bsm_price": "قیمت بلک-شولز",
            "bsm_gap_pct": "اختلاف درصدی بازار با بلک-شولز",

            "bid_price": "بهترین قیمت خرید",
            "bid_volume": "حجم بهترین خرید",
            "ask_price": "بهترین قیمت فروش",
            "ask_volume": "حجم بهترین فروش",

            "open_positions": "موقعیت باز",
            "contract_size": "اندازه قرارداد",
            "notional_value": "ارزش اسمی",

            "trades_num": "تعداد معاملات",
            "trades_volume": "حجم معاملات",
            "trades_value": "ارزش معاملات",

            "yesterday_open_positions": "موقعیت باز دیروز",

            "begin_date": "تاریخ شروع (میلادی خام)",
            "end_date": "تاریخ سررسید (میلادی خام)",

            "وضعیت": "وضعیت قرارداد",
        }

        fa_fallback = {
            "name": "نام قرارداد",
            "option_type": "نوع اختیار (خام)",
        }

        for ccol in list(filtered.columns):
            if ccol not in col_map:
                # اینجا اگر دوست داشتی برای هر ستون جدید، معادل فارسی اضافه کن
                col_map[ccol] = fa_fallback.get(ccol, f"ستون: {ccol}")

        present = list(col_map.values())

        default_cols = [
            "نماد آپشن",
            "نوع اختیار",
            "قیمت اعمال",
            "وضعیت قرارداد",
            "قیمت پریمیوم (آخرین)",
            "قیمت پریمیوم (پایانی)",
            "درصد تغییر پریمیوم نسبت به دیروز",
            "موقعیت باز",
            "قیمت بلک-شولز",
            "اختلاف درصدی بازار با بلک-شولز",
            "سررسید (شمسی)",
            "مانده تا سررسید (روز)",
            "تعداد معاملات",
            "حجم معاملات",
            "ارزش معاملات",
        ]

        cb_cols = st.columns(4)
        picked_cols = []
        for i, fa in enumerate(present):
            key = f"chk_global_{i}_{fa}"
            default_on = fa in default_cols
            with cb_cols[i % 4]:
                if st.checkbox(fa, value=st.session_state.get(key, default_on), key=key):
                    picked_cols.append(fa)

        if not picked_cols:
            picked_cols = [c for c in default_cols if c in present]

        initial_order = [c for c in default_cols if c in picked_cols] + [c for c in picked_cols if c not in default_cols]
        order_key = "order_cols_global"
        sanitize_multiselect_state(order_key, picked_cols, initial_order)

        ordered_cols = st.multiselect(
            "ترتیب نمایش ستون‌ها (از اول تا آخر کلیک کن)",
            options=picked_cols,
            default=st.session_state.get(order_key, initial_order),
            key=order_key,
        )
        if ordered_cols:
            picked_cols = ordered_cols

    # -----------------------------
    # Chain table
    # -----------------------------
    st.markdown("---")
    st.markdown("### جدول زنجیره آپشن‌ها")

    if len(filtered) == 0:
        st.warning("بعد از اعمال فیلترها، هیچ قراردادی باقی نماند.")
        st.session_state["selected_row"] = None
        st.stop()

    def color_status(s):
        if s == "ITM":
            return '<span style="padding:2px 8px;border-radius:999px;background:#e8f5e9;color:#1b5e20;font-weight:800;">ITM</span>'
        if s == "ATM":
            return '<span style="padding:2px 8px;border-radius:999px;background:#fff3e0;color:#e65100;font-weight:800;">ATM</span>'
        if s == "OTM":
            return '<span style="padding:2px 8px;border-radius:999px;background:#ffebee;color:#b71c1c;font-weight:800;">OTM</span>'
        return '<span style="padding:2px 8px;border-radius:999px;background:#eeeeee;color:#424242;font-weight:800;">نامشخص</span>'

    def color_pct(x):
        if x is None or _is_na(x):
            return "—"
        try:
            v = float(x)
            if not np.isfinite(v):
                return "—"
            if v > 0:
                return f'<span class="num-ltr" style="color:#1b5e20;font-weight:800;">{v:,.2f}%</span>'
            if v < 0:
                return f'<span class="num-ltr" style="color:#b71c1c;font-weight:800;">{v:,.2f}%</span>'
            return f'<span class="num-ltr" style="color:#424242;font-weight:800;">{v:,.2f}%</span>'
        except Exception:
            return "—"

    def color_gap(x):
        if x is None or _is_na(x):
            return "—"
        try:
            v = float(x)
            if not np.isfinite(v):
                return "—"
            # gap مثبت یعنی بازار بالاتر از BSM (قرمز) / gap منفی یعنی ارزان‌تر از BSM (سبز)
            if v > 0:
                return f'<span class="num-ltr" style="color:#b71c1c;font-weight:900;">{v:,.2f}%</span>'
            if v < 0:
                return f'<span class="num-ltr" style="color:#1b5e20;font-weight:900;">{v:,.2f}%</span>'
            return f'<span class="num-ltr" style="color:#424242;font-weight:900;">{v:,.2f}%</span>'
        except Exception:
            return "—"

    inv_map = {v: k for k, v in col_map.items()}  # fa -> src
    chain = pd.DataFrame()

    for fa in picked_cols:
        src = inv_map.get(fa)

        if fa == "نماد آپشن":
            chain[fa] = filtered["ticker"]

        elif fa == "نوع اختیار":
            chain[fa] = filtered["نوع اختیار"]

        elif fa == "سررسید (شمسی)":
            chain[fa] = filtered["سررسید (شمسی)"]

        elif fa == "سررسید (میلادی)":
            chain[fa] = filtered["سررسید (میلادی)"]

        elif fa == "وضعیت قرارداد":
            chain[fa] = filtered["وضعیت"].apply(color_status) if "وضعیت" in filtered.columns else color_status("نامشخص")

        elif fa == "درصد تغییر پریمیوم نسبت به دیروز":
            chain[fa] = filtered["pct_change_vs_yesterday"].apply(color_pct)

        elif fa == "اختلاف درصدی بازار با بلک-شولز":
            chain[fa] = filtered["bsm_gap_pct"].apply(color_gap)

        elif fa == "قیمت بلک-شولز":
            chain[fa] = filtered["bsm_price"].apply(
                lambda v: fmt_num(int(round(float(v)))) if (v is not None and not _is_na(v) and np.isfinite(float(v))) else "—"
            )

        elif src in filtered.columns:
            chain[fa] = filtered[src]

        else:
            chain[fa] = "—"

    exclude_fmt = {
        "نماد آپشن",
        "نوع اختیار",
        "سررسید (شمسی)",
        "سررسید (میلادی)",
        "وضعیت قرارداد",
        "درصد تغییر پریمیوم نسبت به دیروز",
        "قیمت بلک-شولز",
        "اختلاف درصدی بازار با بلک-شولز",
    }
    for col in chain.columns:
        if col not in exclude_fmt:
            chain[col] = chain[col].apply(lambda x: wrap_num_ltr(fmt_num(to_int(x))))

    render_table_html(chain, height_px=560)

    st.download_button(
        "⬇️ دانلود CSV جدول زنجیره",
        data=chain.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"chain_{ua_ticker}.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Contract selection + details
    # -----------------------------
    st.markdown("---")

    with st.expander("🧾 انتخاب قرارداد و نمایش اطلاعات", expanded=False):

        def option_label(row):
            t = safe_str(row.get("ticker"))
            typ = safe_str(row.get("نوع اختیار"))
            j = safe_str(row.get("سررسید (شمسی)"))
            k = fmt_num(to_int(row.get("strike_price")))
            dtm = fmt_num(to_int(row.get("days_to_maturity")))
            lp = fmt_num(to_int(row.get("last_price")))
            stt = safe_str(row.get("وضعیت")) if "وضعیت" in row.index else "—"
            return f"{t} | {typ} | {stt} | اعمال: {k} | سررسید: {j} | مانده: {dtm} | پریمیوم: {lp}"

        labels = filtered.apply(option_label, axis=1).tolist()

        sel_key = "sel_contract"
        prev_sel = st.session_state.get(sel_key, 0)
        if isinstance(prev_sel, int) and prev_sel >= len(labels):
            st.session_state[sel_key] = 0

        idx = st.selectbox("قرارداد موردنظر", range(len(labels)), format_func=lambda i: labels[i], key=sel_key)
        selected = filtered.iloc[int(idx)].copy()
        st.session_state["selected_row"] = selected.to_dict()

        details = {
            "نماد آپشن": safe_str(selected.get("ticker")),
            "دارایی پایه": safe_str(selected.get("ua_ticker")),
            "نوع اختیار": safe_str(selected.get("نوع اختیار")),
            "وضعیت قرارداد": safe_str(selected.get("وضعیت")) if "وضعیت" in selected.index else "—",
            "سررسید (شمسی)": safe_str(selected.get("سررسید (شمسی)")),
            "سررسید (میلادی)": safe_str(selected.get("سررسید (میلادی)")),
            "مانده تا سررسید (روز)": to_int(selected.get("days_to_maturity")),
            "قیمت اعمال": to_int(selected.get("strike_price")),

            "قیمت پریمیوم (آخرین)": to_int(selected.get("last_price")),
            "قیمت پریمیوم (پایانی)": to_int(selected.get("close_price")),
            "قیمت پریمیوم (دیروز)": to_int(selected.get("yesterday_price")),
            "درصد تغییر پریمیوم نسبت به دیروز": to_float(selected.get("pct_change_vs_yesterday")),

            "قیمت بلک-شولز": to_float(selected.get("bsm_price")),
            "اختلاف درصدی بازار با بلک-شولز": to_float(selected.get("bsm_gap_pct")),

            "بهترین قیمت خرید": to_int(selected.get("bid_price")),
            "حجم بهترین خرید": to_int(selected.get("bid_volume")),
            "بهترین قیمت فروش": to_int(selected.get("ask_price")),
            "حجم بهترین فروش": to_int(selected.get("ask_volume")),

            "موقعیت باز": to_int(selected.get("open_positions")),
            "اندازه قرارداد": to_int(selected.get("contract_size")),
            "کد TSE": safe_str(selected.get("tse_code")),
        }

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("پریمیوم (آخرین)", fmt_num(details["قیمت پریمیوم (آخرین)"]))
        m2.metric("قیمت اعمال", fmt_num(details["قیمت اعمال"]))
        m3.metric("مانده", fmt_num(details["مانده تا سررسید (روز)"]))
        m4.metric("موقعیت باز", fmt_num(details["موقعیت باز"]))
        m5.metric("وضعیت", details["وضعیت قرارداد"])

        det_all = list(details.keys())
        det_key = "det_cols_global"
        sanitize_multiselect_state(det_key, det_all, det_all)
        det_picked = st.multiselect("ستون‌های جدول جزئیات قرارداد", det_all, key=det_key)

        det_rows = []
        for k in det_picked:
            v = details[k]
            if k in ["درصد تغییر پریمیوم نسبت به دیروز", "اختلاف درصدی بازار با بلک-شولز"]:
                if v is None or _is_na(v):
                    det_rows.append({"عنوان": k, "مقدار": "—"})
                else:
                    det_rows.append({"عنوان": k, "مقدار": f"<span class='num-ltr'>{float(v):,.2f}%</span>"})
            elif k == "قیمت بلک-شولز":
                if v is None or _is_na(v):
                    det_rows.append({"عنوان": k, "مقدار": "—"})
                else:
                    det_rows.append({"عنوان": k, "مقدار": fmt_num(int(round(float(v))))})
            else:
                det_rows.append({"عنوان": k, "مقدار": (v if isinstance(v, str) else wrap_num_ltr(fmt_num(v)))})

        det_df = pd.DataFrame(det_rows)
        render_table_html(det_df, height_px=420, title="جزئیات قرارداد انتخاب‌شده")



# ============================================================
# TAB 2: BSM
# ============================================================
with tab_bsm:
    st.subheader("🧮 بلک-شولز + IV + Greeks")

    row = st.session_state.get("selected_row")
    if not row:
        st.info("اول در تب فیلترها یک قرارداد انتخاب کن.")
        st.stop()

    opt_ticker = safe_str(row.get("ticker"))
    opt_type_fa = safe_str(row.get("نوع اختیار"))
    opt_type = "call" if "خرید" in opt_type_fa else "put"

    K = to_float(row.get("strike_price"))
    days = to_float(row.get("days_to_maturity"))
    S_market = ua_last if ua_last is not None else to_float(row.get("ua_last_price"))

    if any(v is None for v in [S_market, K, days]) or S_market <= 0 or K <= 0 or days <= 0:
        st.error("S یا K یا مانده تا سررسید معتبر نیست.")
        st.stop()

    st.markdown(
        f"**قرارداد:** {opt_ticker}  |  **نوع:** {opt_type_fa}  |  **قیمت اعمال:** {fmt_num(to_int(K))}  |  **مانده:** {fmt_num(to_int(days))} روز"
    )

    prem_choice = st.selectbox("پرمیوم بازار برای IV", ["آخرین", "پایانی"], index=0, key="bsm_prem_choice")
    market_premium = to_float(row.get("last_price")) if prem_choice == "آخرین" else to_float(row.get("close_price"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        r = st.number_input("نرخ بهره r", 0.0, 2.0, 0.30, 0.01, key="bsm_r")
    with c2:
        q = st.number_input("سود نقدی q", 0.0, 2.0, 0.00, 0.01, key="bsm_q")
    with c3:
        sigma = st.number_input("نوسان σ", 0.0001, 5.0, 0.60, 0.05, key="bsm_sigma")
    with c4:
        basis = st.selectbox("مبنای روز/سال", ["365", "252"], index=0, key="bsm_basis")

    T = float(days) / (365.0 if basis == "365" else 252.0)

    # --- قیمت نظری با S بازار
    res_mkt = bsm_price_greeks(S_market, K, T, float(r), float(sigma), option_type=opt_type, q=float(q))
    iv = (
        implied_vol_bisection(market_premium, S_market, K, T, float(r), option_type=opt_type, q=float(q))
        if (market_premium and market_premium > 0)
        else None
    )

    st.markdown("---")
    st.markdown("### 📌 خروجی بر اساس قیمت فعلی دارایی پایه (S بازار)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("S بازار", fmt_num(to_int(S_market)))
    m2.metric("قیمت نظری (BSM)", fmt_num(res_mkt["قیمت نظری (بلک-شولز)"], 4) if res_mkt else "—")
    m3.metric("پرمیوم بازار", fmt_num(market_premium, 4) if market_premium is not None else "—")
    m4.metric("IV", fmt_num(iv, 4) if iv is not None else "—")

    if res_mkt:
        gdf = pd.DataFrame(
            [
                ["دلتا", res_mkt["دلتا"]],
                ["گاما", res_mkt["گاما"]],
                ["وگا", res_mkt["وگا"]],
                ["تتا", res_mkt["تتا"]],
                ["رو", res_mkt["رو"]],
            ],
            columns=["شاخص", "مقدار"],
        )
        gdf["مقدار"] = gdf["مقدار"].apply(lambda x: fmt_num(x, 6))
        render_table_html(gdf, height_px=300, title="Greeks (بر اساس S بازار)")

    # --- سناریوی قیمت پیش‌بینی‌شده
    st.markdown("---")
    st.markdown("### 🔮 سناریوی قیمت پیش‌بینی‌شده دارایی پایه")

    with st.expander("➕ محاسبه بلک-شولز بر اساس S پیش‌بینی‌شده", expanded=True):
        # حداقل و حداکثر پیشنهادی برای input (نسبت به S فعلی)
        s_min = max(1.0, float(S_market) * 0.2)
        s_max = float(S_market) * 5.0

        S_pred = st.number_input(
            "قیمت پیش‌بینی‌شده دارایی پایه (S پیش‌بینی)",
            min_value=float(s_min),
            max_value=float(s_max),
            value=float(st.session_state.get("bsm_S_pred", float(S_market))),
            step=max(1.0, float(S_market) * 0.01),
            key="bsm_S_pred",
        )

        res_pred = bsm_price_greeks(float(S_pred), K, T, float(r), float(sigma), option_type=opt_type, q=float(q))

        p_mkt = res_mkt["قیمت نظری (بلک-شولز)"] if res_mkt else None
        p_pred = res_pred["قیمت نظری (بلک-شولز)"] if res_pred else None
        delta_price = (p_pred - p_mkt) if (p_pred is not None and p_mkt is not None) else None

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("S پیش‌بینی", fmt_num(to_int(S_pred)))
        s2.metric("قیمت نظری (BSM) با S پیش‌بینی", fmt_num(p_pred, 4) if p_pred is not None else "—")
        s3.metric("تغییر قیمت نظری نسبت به S بازار", fmt_num(delta_price, 4) if delta_price is not None else "—")
        s4.metric("S/K", fmt_num(float(S_pred) / float(K), 4))

        if res_pred:
            gdf2 = pd.DataFrame(
                [
                    ["دلتا", res_pred["دلتا"]],
                    ["گاما", res_pred["گاما"]],
                    ["وگا", res_pred["وگا"]],
                    ["تتا", res_pred["تتا"]],
                    ["رو", res_pred["رو"]],
                ],
                columns=["شاخص", "مقدار"],
            )
            gdf2["مقدار"] = gdf2["مقدار"].apply(lambda x: fmt_num(x, 6))
            render_table_html(gdf2, height_px=300, title="Greeks (بر اساس S پیش‌بینی)")

        # خلاصه‌ی مقایسه‌ای
        comp = pd.DataFrame(
            [
                ["S", S_market, S_pred],
                ["قیمت نظری", p_mkt, p_pred],
                ["دلتا", res_mkt["دلتا"] if res_mkt else None, res_pred["دلتا"] if res_pred else None],
                ["گاما", res_mkt["گاما"] if res_mkt else None, res_pred["گاما"] if res_pred else None],
                ["وگا", res_mkt["وگا"] if res_mkt else None, res_pred["وگا"] if res_pred else None],
                ["تتا", res_mkt["تتا"] if res_mkt else None, res_pred["تتا"] if res_pred else None],
                ["رو", res_mkt["رو"] if res_mkt else None, res_pred["رو"] if res_pred else None],
            ],
            columns=["شاخص", "با S بازار", "با S پیش‌بینی"],
        )

        def _fmt_cell(v, is_price=False):
            if v is None or _is_na(v):
                return "—"
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    return "—"
                if is_price:
                    return fmt_num(fv, 4)
                return fmt_num(fv, 6)
            except Exception:
                return safe_str(v)

        comp["با S بازار"] = comp.apply(lambda r: _fmt_cell(r["با S بازار"], is_price=(r["شاخص"] in ["S", "قیمت نظری"])), axis=1)
        comp["با S پیش‌بینی"] = comp.apply(lambda r: _fmt_cell(r["با S پیش‌بینی"], is_price=(r["شاخص"] in ["S", "قیمت نظری"])), axis=1)
        render_table_html(comp, height_px=340, title="مقایسه سریع: S بازار vs S پیش‌بینی")




# ============================================================
# TAB 3: Strategies
# ============================================================

_bold_re = re.compile(r"\*\*(.+?)\*\*")

def render_rtl_guide(text: str):
        if not text:
            return

        # Convert **bold** -> <b>bold</b>
        html = _bold_re.sub(r"<b>\1</b>", text)

        st.markdown(
            f"""
            <div style="
                direction: rtl;
                text-align: right;
                unicode-bidi: embed;
                white-space: pre-line;  
                line-height: 1.9;
                font-size: 0.95rem;
            ">
            {html}
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_strat:
    st.subheader("🎯 استراتژی‌های معروف آپشن (مقایسه سود/زیان)")

    with st.expander("📘 راهنمای استفاده از این تب", expanded=False):
        render_rtl_guide(
            """
**این تب چه کار می‌کند؟**
- برای دارایی پایه‌ای که انتخاب کردی، تعدادی استراتژی معروف آپشن را می‌سازد و **سود/زیان در سررسید** را بر اساس **قیمت هدف (S هدف)** محاسبه می‌کند.
- سپس **بهترین استراتژی** (بیشترین «سود در S هدف») را نمایش می‌دهد و برای هر استراتژی جدول می‌سازد.

**مراحل استفاده**
1) در تب «فیلترهای پیشرفته»، دارایی پایه را انتخاب کن (لازم نیست قرارداد خاصی را انتخاب کنی).
2) وارد تب «استراتژی‌ها» شو.
3) «پرمیوم برای محاسبه» را انتخاب کن:
   - **Mid (Bid/Ask)**: میانگین Bid و Ask (اگر هر دو موجود باشد) – معمولاً منطقی‌تر برای قیمت‌گذاری.
   - **آخرین** یا **پایانی**: در صورت نبود Bid/Ask یا برای حالت‌های خاص.
4) در بخش «قیمت هدف»، مقدار **S هدف** را وارد کن (قیمت مورد انتظار تو در زمان سررسید).
5) جدول‌ها را ببین:
   - **Top نتایج کلی**: بهترین‌ها در بین همه استراتژی‌ها
   - **جدول هر استراتژی**: بهترین ترکیب‌های همان استراتژی

**نکات مهم**
- این مقایسه «سناریومحور» است (بر اساس S هدف). یعنی احتمال وقوع سناریو در نظر گرفته نمی‌شود.
- کارمزد، مالیات، محدودیت‌های سفارش‌گذاری و لغزش قیمت لحاظ نشده‌اند.
- برای استراتژی‌های Covered، فرض شده **یک واحد قرارداد** پوشش داده می‌شود (بر اساس `contract_size` اگر موجود باشد).
"""
        )

    # برای اجرای استراتژی‌ها به هر دو نوع Call/Put نیاز داریم
    try:
        options_all = chains.options(ua_tse_code=ua_tse_code, option_type="both").copy()
    except Exception:
        options_all = options_df.copy()

    if options_all is None or len(options_all) == 0:
        st.warning("داده‌ای برای استراتژی‌ها موجود نیست.")
        st.stop()

    # استانداردسازی ستون‌های لازم
    for c in ["ticker", "tse_code", "ua_ticker", "ua_tse_code"]:
        if c in options_all.columns:
            options_all[c] = options_all[c].astype(str)

    if "نوع اختیار" not in options_all.columns:
        options_all["نوع اختیار"] = options_all["option_type"].apply(map_option_type) if "option_type" in options_all.columns else "—"
    if "سررسید (میلادی)" not in options_all.columns and "end_date" in options_all.columns:
        options_all["سررسید (میلادی)"] = options_all["end_date"].apply(fmt_date_yyyymmdd)
    if "سررسید (شمسی)" not in options_all.columns and "end_date" in options_all.columns:
        options_all["سررسید (شمسی)"] = options_all["end_date"].apply(gregorian_yyyymmdd_to_jalali_str)

    # قیمت مبنا (S فعلی)
    S0 = ua_last if ua_last is not None else to_float(options_all.get("ua_last_price", pd.Series([np.nan])).iloc[0])
    if S0 is None or _is_na(S0) or float(S0) <= 0:
        st.error("قیمت دارایی پایه (S) معتبر نیست.")
        st.stop()
    S0 = float(S0)

    # انتخاب پرمیوم برای محاسبه
    c1, c2, c3 = st.columns([1.2, 1.0, 1.8])
    with c1:
        prem_mode = st.selectbox("پرمیوم برای محاسبه", ["Mid (Bid/Ask)", "آخرین", "پایانی"], index=0, key="str_prem_mode")
    with c2:
        top_n = st.number_input("Top N هر جدول", 5, 50, 10, 1, key="str_topn")
    with c3:
        st.caption("⚠️ این مقایسه «بر اساس قیمت هدف/سناریو» انجام می‌شود (نه پیش‌بینی احتمالی).")

    def get_premium(row):
        bid = to_float(row.get("bid_price"))
        ask = to_float(row.get("ask_price"))
        lastp = to_float(row.get("last_price"))
        closep = to_float(row.get("close_price"))
        if prem_mode.startswith("Mid"):
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                return 0.5 * (bid + ask)
            # fallback
            return lastp if (lastp is not None and lastp > 0) else closep
        if prem_mode == "آخرین":
            return lastp if (lastp is not None and lastp > 0) else closep
        return closep if (closep is not None and closep > 0) else lastp

    # ورودی سناریو: قیمت هدف
    with st.expander("🔮 سناریو / قیمت هدف دارایی پایه", expanded=True):
        s_min = max(1.0, S0 * 0.2)
        s_max = S0 * 5.0
        S_target = st.number_input(
            "قیمت هدف (S هدف)",
            min_value=float(s_min),
            max_value=float(s_max),
            value=float(st.session_state.get("bsm_S_pred", S0)),  # از تب بلک‌شولز هم استفاده می‌کنیم
            step=max(1.0, S0 * 0.01),
            key="str_S_target",
        )
        S_target = float(S_target)

        st.markdown(
            f"قیمت فعلی: <span class='num-ltr'><b>{S0:,.0f}</b></span>  |  "
            f"قیمت هدف: <span class='num-ltr'><b>{S_target:,.0f}</b></span>",
            unsafe_allow_html=True,
        )

    # دیتای پایه
    df = options_all.copy()
    for col in ["strike_price", "days_to_maturity", "last_price", "close_price", "bid_price", "ask_price", "contract_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["premium"] = df.apply(get_premium, axis=1)
    df = df[df["premium"].notna() & (df["premium"] > 0)]
    df = df[df["strike_price"].notna() & (df["strike_price"] > 0)]
    df = df[df["days_to_maturity"].notna() & (df["days_to_maturity"] > 0)]

    # انتخاب سررسیدها
    mats = sorted([m for m in df["سررسید (شمسی)"].dropna().unique().tolist() if m != "—"])
    chosen_mats = st.multiselect("سررسید (شمسی) برای محاسبه", mats, default=mats[: min(3, len(mats))], key="str_mats")
    if chosen_mats:
        df = df[df["سررسید (شمسی)"].isin(chosen_mats)]

    if len(df) == 0:
        st.warning("بعد از فیلترها داده‌ای باقی نماند.")
        st.stop()

    # محدودسازی برای جلوگیری از انفجار ترکیبی
    def select_near_atm(sub: pd.DataFrame, kcol="strike_price", n=20):
        sub = sub.copy()
        sub["dist_atm"] = (sub[kcol] - S0).abs()
        return sub.sort_values("dist_atm").head(n).drop(columns=["dist_atm"])


    def legs_html(lines):
        """Readable legs (RTL) with clear Buy/Sell; returns HTML with <br>."""
        if not lines:
            return "—"
        return "<br>".join([f"• {ln}" for ln in lines])

    def style_profit_dataframe(df_in: pd.DataFrame, profit_col: str):
        """Return a pandas Styler coloring profit column; Streamlit dataframe remains sortable."""
        df2 = df_in.copy()
        if profit_col in df2.columns:
            df2[profit_col] = pd.to_numeric(df2[profit_col], errors="coerce")

            def _sty(v):
                try:
                    if v > 0:
                        return "color:#1b5e20;font-weight:900"
                    if v < 0:
                        return "color:#b71c1c;font-weight:900"
                    return "color:#424242"
                except Exception:
                    return ""

            return df2.style.applymap(_sty, subset=[profit_col]).format(precision=2, na_rep="—")
        return df2


    def _fmt_num_html(v, decimals=2, is_pct=False):
        try:
            if v is None or _is_na(v):
                return "—"
            fv = float(v)
            if not np.isfinite(fv):
                return "—"
            s = f"{fv:,.{decimals}f}"
            if is_pct:
                s += "%"
            return wrap_num_ltr(s)
        except Exception:
            return safe_str(v)

    def _profit_html(v):
        try:
            if v is None or _is_na(v):
                return "—"
            fv = float(v)
            if not np.isfinite(fv):
                return "—"
            if fv > 0:
                col = "#1b5e20"
            elif fv < 0:
                col = "#b71c1c"
            else:
                col = "#424242"
            return f"<span class='num-ltr' style='color:{col};font-weight:900'>{fv:,.2f}</span>"
        except Exception:
            return safe_str(v)

    def show_table(df_in: pd.DataFrame, title: str, key_prefix: str, top_n: int):
        """Sortable (server-side) + full visibility via HTML table (no truncation)."""
        if df_in is None or len(df_in) == 0:
            st.info("داده‌ای برای نمایش وجود ندارد.")
            return

        st.markdown(f"#### {title}")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sort_col = st.selectbox("مرتب‌سازی بر اساس", df_in.columns.tolist(), index=0, key=f"{key_prefix}_sort")
        with c2:
            sort_asc = st.checkbox("صعودی", value=False, key=f"{key_prefix}_asc")
        with c3:
            n = st.number_input("تعداد ردیف", 5, 200, int(top_n), 5, key=f"{key_prefix}_n")

        dfx = df_in.copy()

        try:
            dfx["_sort_tmp"] = pd.to_numeric(dfx[sort_col], errors="coerce")
            if dfx["_sort_tmp"].notna().any():
                dfx = dfx.sort_values("_sort_tmp", ascending=sort_asc)
            else:
                dfx = dfx.sort_values(sort_col, ascending=sort_asc)
            dfx = dfx.drop(columns=["_sort_tmp"], errors="ignore")
        except Exception:
            pass

        dfx = dfx.head(int(n)).copy()

        # HTML-friendly formatting
        if "سود در S هدف" in dfx.columns:
            dfx["سود در S هدف"] = pd.to_numeric(dfx["سود در S هدف"], errors="coerce").apply(_profit_html)
        if "ROI% تقریبی" in dfx.columns:
            dfx["ROI% تقریبی"] = pd.to_numeric(dfx["ROI% تقریبی"], errors="coerce").apply(lambda x: _fmt_num_html(x, 2, True))
        for col in ["هزینه/دریافتی (پرمیوم خالص)", "حداکثر زیان"]:
            if col in dfx.columns:
                dfx[col] = pd.to_numeric(dfx[col], errors="coerce").apply(lambda x: _fmt_num_html(x, 0, False))
        if "حداکثر سود" in dfx.columns:
            def _mx(v):
                if isinstance(v, str):
                    return v
                return _fmt_num_html(v, 0, False)
            dfx["حداکثر سود"] = dfx["حداکثر سود"].apply(_mx)

        render_table_html(dfx, height_px=460, title=None)

    # محاسبات استراتژی‌ها (سود در سررسید)
    def payoff_long_call(S, K, prem): return max(S - K, 0.0) - prem
    def payoff_long_put(S, K, prem): return max(K - S, 0.0) - prem

    def make_row(strategy, mat, legs, net_premium, max_profit, max_loss, breakeven, profit_at_target, roi, days):
        return {
            "استراتژی": strategy,
            "سررسید": mat,
            "مانده (روز)": days,
            "لِگ‌ها": legs,
            "هزینه/دریافتی (پرمیوم خالص)": net_premium,
            "حداکثر سود": max_profit,
            "حداکثر زیان": max_loss,
            "نقطه/نقاط سربه‌سر": breakeven,
            "سود در S هدف": profit_at_target,
            "ROI% تقریبی": roi,
        }

    rows = []

    # گروه‌بندی بر اساس سررسید
    for mat, g in df.groupby("سررسید (شمسی)"):
        days_left = int(pd.to_numeric(g["days_to_maturity"], errors="coerce").min())
        calls = g[g["نوع اختیار"].astype(str).str.contains("خرید", na=False)].copy()
        puts = g[g["نوع اختیار"].astype(str).str.contains("فروش", na=False)].copy()

        # نزدیک به ATM برای ترکیبی‌ها
        calls_n = select_near_atm(calls, n=20) if len(calls) else calls
        puts_n = select_near_atm(puts, n=20) if len(puts) else puts

        # --- Long Call / Long Put (تک‌قرارداد)
        for _, r0 in calls.iterrows():
            K = float(r0["strike_price"])
            prem = float(r0["premium"])
            pr = payoff_long_call(S_target, K, prem)
            legs = legs_html([
                f"خرید Call: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
            ])
            max_loss = -prem
            max_profit = "نامحدود"
            breakeven = f"{(K + prem):,.0f}"
            roi = (pr / prem * 100.0) if prem > 0 else np.nan
            rows.append(make_row("Long Call", mat, legs, prem, max_profit, max_loss, breakeven, pr, roi, days_left))

        for _, r0 in puts.iterrows():
            K = float(r0["strike_price"])
            prem = float(r0["premium"])
            pr = payoff_long_put(S_target, K, prem)
            legs = legs_html([
                f"خرید Put: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
            ])
            max_loss = -prem
            max_profit = f"{(K - prem):,.0f}"  # تقریبی (تا S=0)
            breakeven = f"{(K - prem):,.0f}"
            roi = (pr / prem * 100.0) if prem > 0 else np.nan
            rows.append(make_row("Long Put", mat, legs, prem, max_profit, max_loss, breakeven, pr, roi, days_left))


        
        # --- Short Call / Short Put (تک‌قرارداد)
        for _, r0 in calls.iterrows():
            K = float(r0["strike_price"])
            prem = float(r0["premium"])  # premium received
            pr = prem - max(S_target - K, 0.0)
            legs = legs_html([
                f"فروش Call: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
            ])
            max_profit = prem
            max_loss = "نامحدود"
            breakeven = f"{(K + prem):,.0f}"
            roi = np.nan  # مارجین/وجه تضمین در این محاسبه لحاظ نشده
            rows.append(make_row("Short Call", mat, legs, -prem, max_profit, max_loss, breakeven, pr, roi, days_left))

        for _, r0 in puts.iterrows():
            K = float(r0["strike_price"])
            prem = float(r0["premium"])  # premium received
            pr = prem - max(K - S_target, 0.0)
            legs = legs_html([
                f"فروش Put: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
            ])
            max_profit = prem
            max_loss = f"{(prem - K):,.0f}"  # تا S=0
            breakeven = f"{(K - prem):,.0f}"
            roi = np.nan  # وجه تضمین لحاظ نشده
            rows.append(make_row("Short Put", mat, legs, -prem, max_profit, max_loss, breakeven, pr, roi, days_left))

# --- Covered Call: Long Underlying + Short Call (فرض: پوشش یک قرارداد)
        if len(calls) > 0:
            # تعداد سهم برای پوشش: از contract_size استفاده می‌کنیم، اگر نبود 1
            for _, r0 in calls.iterrows():
                K = float(r0["strike_price"])
                prem = float(r0["premium"])  # premium received from selling call
                csize = to_float(r0.get("contract_size"))
                csize = float(csize) if (csize is not None and not _is_na(csize) and csize > 0) else 1.0

                # Payoff per ONE underlying unit:
                # (S_T - S0) + prem - max(S_T-K,0)
                pr_unit = (S_target - S0) + prem - max(S_target - K, 0.0)
                pr = pr_unit * csize

                # Risk/return:
                # max profit when S_T >= K: (K - S0) + prem
                max_profit = ((K - S0) + prem) * csize
                # max loss roughly when S_T -> 0: (-S0 + prem) * csize
                max_loss = ((0.0 - S0) + prem) * csize

                breakeven = (S0 - prem)
                roi = (pr / (S0 * csize) * 100.0) if (S0 * csize) > 0 else np.nan

                legs = legs_html([
                    f"خرید سهم پایه: {wrap_num_ltr(f'{csize:,.0f}')} واحد",
                    f"فروش Call: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
                ])
                rows.append(make_row("Covered Call", mat, legs, -prem * csize, max_profit, max_loss, f"{breakeven:,.0f}", pr, roi, days_left))

        # --- Cash-Secured Put: Short Put (پول نقد برای خرید سهم در K)
        if len(puts) > 0:
            for _, r0 in puts.iterrows():
                K = float(r0["strike_price"])
                prem = float(r0["premium"])  # premium received
                csize = to_float(r0.get("contract_size"))
                csize = float(csize) if (csize is not None and not _is_na(csize) and csize > 0) else 1.0

                # payoff per unit: prem - max(K - S_T, 0)
                pr_unit = prem - max(K - S_target, 0.0)
                pr = pr_unit * csize

                # max profit: prem
                max_profit = prem * csize
                # max loss at S_T -> 0: prem - K
                max_loss = (prem - K) * csize
                breakeven = (K - prem)
                cash_req = K * csize
                roi = (pr / cash_req * 100.0) if cash_req > 0 else np.nan

                legs = legs_html([
                    f"فروش Put (Cash-Secured): <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
                ])
                rows.append(make_row("Short Put", mat, legs, -prem * csize, max_profit, max_loss, f"{breakeven:,.0f}", pr, roi, days_left))

        
        # --- Protective Put: Long Underlying + Long Put
        if len(puts) > 0:
            for _, r0 in puts.iterrows():
                K = float(r0["strike_price"])
                prem = float(r0["premium"])  # premium paid
                csize = to_float(r0.get("contract_size"))
                csize = float(csize) if (csize is not None and not _is_na(csize) and csize > 0) else 1.0

                # (S_T - S0) + max(K - S_T,0) - prem
                pr_unit = (S_target - S0) + max(K - S_target, 0.0) - prem
                pr = pr_unit * csize

                max_profit = "نامحدود"
                max_loss = ((K - S0) - prem) * csize  # در بدترین حالت
                breakeven = (S0 + prem)
                roi = (pr / (S0 * csize) * 100.0) if (S0 * csize) > 0 else np.nan

                legs = legs_html([
                    f"خرید سهم پایه: {wrap_num_ltr(f'{csize:,.0f}')} واحد",
                    f"خرید Put: <b>{r0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem:,.0f}')}",
                ])
                rows.append(make_row("Protective Put", mat, legs, prem * csize, max_profit, max_loss, f"{breakeven:,.0f}", pr, roi, days_left))

        # --- Collar: Long Underlying + Long Put (OTM) + Short Call (OTM)
        if len(calls_n) and len(puts_n):
            c_otm = calls_n[calls_n["strike_price"].astype(float) >= S0].sort_values("strike_price").head(8)
            p_otm = puts_n[puts_n["strike_price"].astype(float) <= S0].sort_values("strike_price", ascending=False).head(8)
            for _, pr0 in p_otm.iterrows():
                Kp = float(pr0["strike_price"])
                prem_p = float(pr0["premium"])
                for _, cr in c_otm.iterrows():
                    Kc = float(cr["strike_price"])
                    prem_c = float(cr["premium"])  # received
                    if Kp >= Kc:
                        continue
                    net = prem_p - prem_c  # debit(+)/credit(-)
                    # payoff: (S-S0) + max(Kp-S,0) - prem_p + prem_c - max(S-Kc,0)
                    pr_val = (S_target - S0) + max(Kp - S_target, 0.0) - prem_p + prem_c - max(S_target - Kc, 0.0)

                    max_profit = ((Kc - S0) - net)
                    max_loss = ((Kp - S0) - net)
                    breakeven = (S0 + net)
                    roi = np.nan

                    legs = legs_html([
                        f"خرید سهم پایه: 1 واحد",
                        f"خرید Put: <b>{pr0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kp:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                        f"فروش Call: <b>{cr.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kc:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                        f"پرمیوم خالص = {wrap_num_ltr(f'{net:,.0f}')}",
                    ])
                    rows.append(make_row("Collar", mat, legs, net, max_profit, max_loss, f"{breakeven:,.0f}", pr_val, roi, days_left))

        # --- Conversion: Long Underlying + Long Put + Short Call (هم‌استرایک)
        if len(calls_n) and len(puts_n):
            calls_n2 = calls_n.copy()
            puts_n2 = puts_n.copy()
            calls_n2["K"] = calls_n2["strike_price"].astype(float)
            puts_n2["K"] = puts_n2["strike_price"].astype(float)
            common = sorted(set(calls_n2["K"]).intersection(set(puts_n2["K"])))
            common = sorted(common, key=lambda k: abs(k - S0))[:10]
            for K in common:
                c0 = calls_n2[calls_n2["K"] == K].iloc[0].to_dict()
                p0 = puts_n2[puts_n2["K"] == K].iloc[0].to_dict()
                prem_c = float(c0.get("premium") or 0.0)  # received
                prem_p = float(p0.get("premium") or 0.0)  # paid
                net = prem_p - prem_c  # debit(+)/credit(-)
                # payoff ثابت: K - S0 - net
                pr_val = (K - S0) - net
                max_profit = pr_val
                max_loss = pr_val
                breakeven = "—"
                roi = np.nan

                legs = legs_html([
                    "خرید سهم پایه: 1 واحد",
                    f"خرید Put: <b>{p0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                    f"فروش Call: <b>{c0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                    f"پرمیوم خالص = {wrap_num_ltr(f'{net:,.0f}')}",
                ])
                rows.append(make_row("Conversion", mat, legs, net, max_profit, max_loss, breakeven, pr_val, roi, days_left))

# --- Bull Call Spread (Debit): Buy lower K, Sell higher K
        if len(calls_n) >= 2:
            c_sorted = calls_n.sort_values("strike_price")
            c_list = c_sorted.to_dict("records")
            for i in range(len(c_list) - 1):
                for j in range(i + 1, len(c_list)):
                    c_buy, c_sell = c_list[i], c_list[j]
                    K1, K2 = float(c_buy["strike_price"]), float(c_sell["strike_price"])
                    prem1, prem2 = float(c_buy["premium"]), float(c_sell["premium"])
                    net = prem1 - prem2  # debit
                    width = K2 - K1
                    if width <= 0:
                        continue
                    payoff = min(max(S_target - K1, 0.0), width) - net
                    max_profit = width - net
                    max_loss = -net
                    breakeven = K1 + net
                    roi = (payoff / abs(net) * 100.0) if net != 0 else np.nan
                    legs = legs_html([
                        f"خرید Call: <b>{c_buy.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem1:,.0f}')}",
                        f"فروش Call: <b>{c_sell.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem2:,.0f}')}",
                    ])
                    rows.append(make_row("Bull Call Spread", mat, legs, net, max_profit, max_loss, f"{breakeven:,.0f}", payoff, roi, days_left))

        
        # --- Bear Call Spread (Credit): Sell lower K, Buy higher K
        if len(calls_n) >= 2:
            c_sorted = calls_n.sort_values("strike_price")
            c_list = c_sorted.to_dict("records")
            for i in range(len(c_list) - 1):
                for j in range(i + 1, len(c_list)):
                    c_sell, c_buy = c_list[i], c_list[j]  # sell lower, buy higher
                    K1, K2 = float(c_sell["strike_price"]), float(c_buy["strike_price"])
                    prem_sell, prem_buy = float(c_sell["premium"]), float(c_buy["premium"])
                    width = K2 - K1
                    if width <= 0:
                        continue
                    credit = prem_sell - prem_buy
                    payoff = credit - min(max(S_target - K1, 0.0), width)
                    max_profit = credit
                    max_loss = (credit - width)
                    breakeven = K1 + credit
                    roi = np.nan
                    legs = legs_html([
                        f"فروش Call: <b>{c_sell.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_sell:,.0f}')}",
                        f"خرید Call: <b>{c_buy.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_buy:,.0f}')}",
                        f"اعتبار (Credit) = {wrap_num_ltr(f'{credit:,.0f}')}",
                    ])
                    rows.append(make_row("Bear Call Spread", mat, legs, -credit, max_profit, max_loss, f"{breakeven:,.0f}", payoff, roi, days_left))

# --- Bear Put Spread (Debit): Buy higher K, Sell lower K
        if len(puts_n) >= 2:
            p_sorted = puts_n.sort_values("strike_price")
            p_list = p_sorted.to_dict("records")
            for i in range(len(p_list) - 1):
                for j in range(i + 1, len(p_list)):
                    p_low, p_high = p_list[i], p_list[j]
                    # buy higher K, sell lower K
                    buy, sell = p_high, p_low
                    K2, K1 = float(buy["strike_price"]), float(sell["strike_price"])
                    prem_buy, prem_sell = float(buy["premium"]), float(sell["premium"])
                    net = prem_buy - prem_sell  # debit
                    width = K2 - K1
                    if width <= 0:
                        continue
                    payoff = min(max(K2 - S_target, 0.0), width) - net
                    max_profit = width - net
                    max_loss = -net
                    breakeven = K2 - net
                    roi = (payoff / abs(net) * 100.0) if net != 0 else np.nan
                    legs = legs_html([
                        f"خرید Put: <b>{buy.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_buy:,.0f}')}",
                        f"فروش Put: <b>{sell.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_sell:,.0f}')}",
                    ])
                    rows.append(make_row("Bear Put Spread", mat, legs, net, max_profit, max_loss, f"{breakeven:,.0f}", payoff, roi, days_left))

        # --- Straddle (ATM-ish): Buy Call+Put same K (تقریباً)
        if len(calls_n) and len(puts_n):
            # Map by strike (rounded) for matching
            calls_n2 = calls_n.copy()
            puts_n2 = puts_n.copy()
            calls_n2["K"] = calls_n2["strike_price"].astype(float)
            puts_n2["K"] = puts_n2["strike_price"].astype(float)
            # find common strikes
            common = sorted(set(calls_n2["K"]).intersection(set(puts_n2["K"])))
            # prefer near ATM
            common = sorted(common, key=lambda k: abs(k - S0))[:10]
            for K in common:
                c0 = calls_n2[calls_n2["K"] == K].iloc[0].to_dict()
                p0 = puts_n2[puts_n2["K"] == K].iloc[0].to_dict()
                net = float(c0["premium"]) + float(p0["premium"])
                payoff = abs(S_target - K) - net
                max_loss = -net
                max_profit = "نامحدود"
                be1, be2 = K - net, K + net
                roi = (payoff / net * 100.0) if net > 0 else np.nan
                prem_c = float(c0.get("premium") or 0)
                prem_p = float(p0.get("premium") or 0)
                legs = legs_html([
                    f"خرید Call: <b>{c0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                    f"خرید Put: <b>{p0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                    f"مجموع پرمیوم = {wrap_num_ltr(f'{net:,.0f}')}",
                ])
                rows.append(make_row("Long Straddle", mat, legs, net, max_profit, max_loss, f"{be1:,.0f} , {be2:,.0f}", payoff, roi, days_left))

        # --- Strangle: Buy OTM Put (Kp<S0) + Buy OTM Call (Kc>S0)
        if len(calls_n) and len(puts_n):
            c_otm = calls_n[calls_n["strike_price"].astype(float) >= S0].sort_values("strike_price").head(10)
            p_otm = puts_n[puts_n["strike_price"].astype(float) <= S0].sort_values("strike_price", ascending=False).head(10)
            if len(c_otm) and len(p_otm):
                for _, cr in c_otm.iterrows():
                    for _, pr0 in p_otm.iterrows():
                        Kc = float(cr["strike_price"])
                        Kp = float(pr0["strike_price"])
                        net = float(cr["premium"]) + float(pr0["premium"])
                        payoff = max(S_target - Kc, 0.0) + max(Kp - S_target, 0.0) - net
                        max_loss = -net
                        max_profit = "نامحدود"
                        be_low = Kp - net
                        be_high = Kc + net
                        roi = (payoff / net * 100.0) if net > 0 else np.nan
                        prem_p = float(pr0.get("premium") or 0)
                        prem_c = float(cr.get("premium") or 0)
                        legs = legs_html([
                            f"خرید Put: <b>{pr0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kp:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                            f"خرید Call: <b>{cr.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kc:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                            f"مجموع پرمیوم = {wrap_num_ltr(f'{net:,.0f}')}",
                        ])
                        rows.append(make_row("Long Strangle", mat, legs, net, max_profit, max_loss, f"{be_low:,.0f} , {be_high:,.0f}", payoff, roi, days_left))
        
        # --- Short Straddle: Sell Call + Sell Put (same K)
        if len(calls_n) and len(puts_n):
            calls_n2 = calls_n.copy()
            puts_n2 = puts_n.copy()
            calls_n2["K"] = calls_n2["strike_price"].astype(float)
            puts_n2["K"] = puts_n2["strike_price"].astype(float)
            common = sorted(set(calls_n2["K"]).intersection(set(puts_n2["K"])))
            common = sorted(common, key=lambda k: abs(k - S0))[:10]
            for K in common:
                c0 = calls_n2[calls_n2["K"] == K].iloc[0].to_dict()
                p0 = puts_n2[puts_n2["K"] == K].iloc[0].to_dict()
                prem_c = float(c0.get("premium") or 0.0)
                prem_p = float(p0.get("premium") or 0.0)
                credit = prem_c + prem_p
                payoff = credit - abs(S_target - K)
                max_profit = credit
                max_loss = "نامحدود"
                be1, be2 = K - credit, K + credit
                roi = np.nan
                legs = legs_html([
                    f"فروش Call: <b>{c0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                    f"فروش Put: <b>{p0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                    f"مجموع اعتبار = {wrap_num_ltr(f'{credit:,.0f}')}",
                ])
                rows.append(make_row("Short Straddle", mat, legs, -credit, max_profit, max_loss, f"{be1:,.0f} , {be2:,.0f}", payoff, roi, days_left))

        # --- Short Strangle: Sell OTM Put + Sell OTM Call
        if len(calls_n) and len(puts_n):
            c_otm = calls_n[calls_n["strike_price"].astype(float) >= S0].sort_values("strike_price").head(10)
            p_otm = puts_n[puts_n["strike_price"].astype(float) <= S0].sort_values("strike_price", ascending=False).head(10)
            for _, cr in c_otm.iterrows():
                for _, pr0 in p_otm.iterrows():
                    Kc = float(cr["strike_price"])
                    Kp = float(pr0["strike_price"])
                    if Kp >= Kc:
                        continue
                    prem_c = float(cr["premium"])
                    prem_p = float(pr0["premium"])
                    credit = prem_c + prem_p
                    payoff = credit - max(Kp - S_target, 0.0) - max(S_target - Kc, 0.0)
                    max_profit = credit
                    max_loss = "نامحدود"
                    be_low = Kp - credit
                    be_high = Kc + credit
                    roi = np.nan
                    legs = legs_html([
                        f"فروش Put: <b>{pr0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kp:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                        f"فروش Call: <b>{cr.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kc:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                        f"مجموع اعتبار = {wrap_num_ltr(f'{credit:,.0f}')}",
                    ])
                    rows.append(make_row("Short Strangle", mat, legs, -credit, max_profit, max_loss, f"{be_low:,.0f} , {be_high:,.0f}", payoff, roi, days_left))

        # --- Long Gut / Short Gut (ITM Strangle با دو استرایک متفاوت)
        if len(calls_n) and len(puts_n):
            c_itm = calls_n[calls_n["strike_price"].astype(float) <= S0].sort_values("strike_price", ascending=False).head(8)
            p_itm = puts_n[puts_n["strike_price"].astype(float) >= S0].sort_values("strike_price").head(8)
            for _, cr in c_itm.iterrows():
                for _, pr0 in p_itm.iterrows():
                    Kc = float(cr["strike_price"])
                    Kp = float(pr0["strike_price"])
                    if Kc >= Kp:
                        continue
                    prem_c = float(cr["premium"])
                    prem_p = float(pr0["premium"])
                    net = prem_c + prem_p  # debit
                    payoff = max(S_target - Kc, 0.0) + max(Kp - S_target, 0.0) - net
                    be_low = Kp - net
                    be_high = Kc + net
                    legs = legs_html([
                        f"خرید Call (ITM): <b>{cr.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kc:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                        f"خرید Put (ITM): <b>{pr0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kp:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                        f"مجموع پرمیوم = {wrap_num_ltr(f'{net:,.0f}')}",
                    ])
                    rows.append(make_row("Long Gut", mat, legs, net, "نامحدود", -net, f"{be_low:,.0f} , {be_high:,.0f}", payoff, (payoff/net*100.0) if net>0 else np.nan, days_left))

                    credit = net
                    payoff_s = credit - (max(S_target - Kc, 0.0) + max(Kp - S_target, 0.0))
                    legs_s = legs_html([
                        f"فروش Call (ITM): <b>{cr.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kc:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c:,.0f}')}",
                        f"فروش Put (ITM): <b>{pr0.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{Kp:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p:,.0f}')}",
                        f"مجموع اعتبار = {wrap_num_ltr(f'{credit:,.0f}')}",
                    ])
                    rows.append(make_row("Short Gut", mat, legs_s, -credit, credit, "نامحدود", f"{be_low:,.0f} , {be_high:,.0f}", payoff_s, np.nan, days_left))

        # --- Call Ratio Spread (1x2): Buy 1 Call K1, Sell 2 Calls K2 (K2>K1)
        if len(calls_n) >= 2:
            c_sorted = calls_n.sort_values("strike_price")
            c_list = c_sorted.to_dict("records")
            for i in range(len(c_list) - 1):
                for j in range(i + 1, len(c_list)):
                    c_buy, c_sell = c_list[i], c_list[j]
                    K1, K2 = float(c_buy["strike_price"]), float(c_sell["strike_price"])
                    prem_buy, prem_sell = float(c_buy["premium"]), float(c_sell["premium"])
                    if K2 <= K1:
                        continue
                    net = prem_buy - 2.0 * prem_sell  # debit(+)/credit(-)
                    payoff = max(S_target - K1, 0.0) - 2.0 * max(S_target - K2, 0.0) - net

                    # breakevens (تقریبی)
                    be_low = K1 + net
                    be_high = (2.0 * K2 - K1) - net
                    legs = legs_html([
                        f"خرید Call: <b>{c_buy.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_buy:,.0f}')}",
                        f"فروش 2× Call: <b>{c_sell.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_sell:,.0f}')}",
                        f"پرمیوم خالص = {wrap_num_ltr(f'{net:,.0f}')}",
                    ])
                    rows.append(make_row("Call Ratio Spread", mat, legs, net, f"{(K2-K1 - net):,.0f}", "نامحدود", f"{be_low:,.0f} , {be_high:,.0f}", payoff, np.nan, days_left))

        # --- Short Call Butterfly: Sell K1, Buy 2x K2, Sell K3 (Reverse of Long)
        if len(calls_n) >= 3:
            c_sorted = calls_n.sort_values("strike_price")
            c_list = c_sorted.to_dict("records")
            for i in range(len(c_list) - 2):
                for j in range(i + 1, len(c_list) - 1):
                    for k in range(j + 1, len(c_list)):
                        c1, c2, c3 = c_list[i], c_list[j], c_list[k]
                        K1, K2, K3 = float(c1["strike_price"]), float(c2["strike_price"]), float(c3["strike_price"])
                        if not (K1 < K2 < K3):
                            continue
                        prem1, prem2, prem3 = float(c1["premium"]), float(c2["premium"]), float(c3["premium"])
                        credit = prem1 - 2.0 * prem2 + prem3  # sell1 - buy2 + sell1
                        width = min(K2 - K1, K3 - K2)
                        payoff = credit - (min(max(S_target - K1, 0.0), K2 - K1) - min(max(S_target - K2, 0.0), K3 - K2))

                        max_profit = credit
                        max_loss = credit - width
                        be_low = K1 + credit
                        be_high = K3 - credit

                        legs = legs_html([
                            f"فروش Call: <b>{c1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem1:,.0f}')}",
                            f"خرید 2× Call: <b>{c2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem2:,.0f}')}",
                            f"فروش Call: <b>{c3.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K3:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem3:,.0f}')}",
                            f"اعتبار (Credit) = {wrap_num_ltr(f'{credit:,.0f}')}",
                        ])
                        rows.append(make_row("Short Call Butterfly", mat, legs, -credit, max_profit, max_loss, f"{be_low:,.0f} , {be_high:,.0f}", payoff, np.nan, days_left))

        # --- Long Box / Short Box (با 2 استرایک مشترک)
        if len(calls_n) and len(puts_n):
            calls_n2 = calls_n.copy()
            puts_n2 = puts_n.copy()
            calls_n2["K"] = calls_n2["strike_price"].astype(float)
            puts_n2["K"] = puts_n2["strike_price"].astype(float)
            common = sorted(set(calls_n2["K"]).intersection(set(puts_n2["K"])))
            common = sorted(common, key=lambda k: abs(k - S0))[:12]
            common_sorted = sorted(common)
            for a in range(len(common_sorted) - 1):
                for b in range(a + 1, len(common_sorted)):
                    K1, K2 = float(common_sorted[a]), float(common_sorted[b])
                    if K2 <= K1:
                        continue
                    cK1 = calls_n2[calls_n2["K"] == K1].iloc[0].to_dict()
                    cK2 = calls_n2[calls_n2["K"] == K2].iloc[0].to_dict()
                    pK1 = puts_n2[puts_n2["K"] == K1].iloc[0].to_dict()
                    pK2 = puts_n2[puts_n2["K"] == K2].iloc[0].to_dict()

                    prem_c1, prem_c2 = float(cK1.get("premium") or 0.0), float(cK2.get("premium") or 0.0)
                    prem_p1, prem_p2 = float(pK1.get("premium") or 0.0), float(pK2.get("premium") or 0.0)

                    # Long Box: (Buy call spread) + (Buy put spread)
                    cost = (prem_c1 - prem_c2) + (prem_p2 - prem_p1)
                    payoff = (K2 - K1) - cost
                    legs = legs_html([
                        f"خرید Call(K1): <b>{cK1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c1:,.0f}')}",
                        f"فروش Call(K2): <b>{cK2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c2:,.0f}')}",
                        f"خرید Put(K2): <b>{pK2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p2:,.0f}')}",
                        f"فروش Put(K1): <b>{pK1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p1:,.0f}')}",
                        f"هزینه خالص = {wrap_num_ltr(f'{cost:,.0f}')}",
                    ])
                    rows.append(make_row("Long Box", mat, legs, cost, f"{(K2-K1 - cost):,.0f}", f"{(K2-K1 - cost):,.0f}", "—", payoff, np.nan, days_left))

                    # Short Box: دریافت اعتبار و پرداخت (K2-K1) در سررسید
                    credit = -cost
                    payoff_s = credit - (K2 - K1)
                    legs_s = legs_html([
                        f"فروش Call(K1): <b>{cK1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c1:,.0f}')}",
                        f"خرید Call(K2): <b>{cK2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_c2:,.0f}')}",
                        f"فروش Put(K2): <b>{pK2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p2:,.0f}')}",
                        f"خرید Put(K1): <b>{pK1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_p1:,.0f}')}",
                        f"اعتبار خالص = {wrap_num_ltr(f'{credit:,.0f}')}",
                    ])
                    rows.append(make_row("Short Box", mat, legs_s, -credit, payoff_s, payoff_s, "—", payoff_s, np.nan, days_left))

# --- Iron Condor (Short / Credit): Sell OTM Put + Buy lower Put + Sell OTM Call + Buy higher Call
        if len(calls_n) and len(puts_n):
            c_otm = calls_n[calls_n["strike_price"].astype(float) >= S0].sort_values("strike_price").head(8)
            p_otm = puts_n[puts_n["strike_price"].astype(float) <= S0].sort_values("strike_price", ascending=False).head(8)

            p_wings = puts_n.sort_values("strike_price", ascending=False).head(12)
            c_wings = calls_n.sort_values("strike_price").head(12)

            for _, sp in p_otm.iterrows():  # short put
                K2 = float(sp["strike_price"])
                prem_sp = float(sp["premium"])
                lp_cands = p_wings[p_wings["strike_price"].astype(float) < K2].sort_values("strike_price", ascending=False).head(4)
                for _, lp in lp_cands.iterrows():
                    K1 = float(lp["strike_price"])
                    prem_lp = float(lp["premium"])

                    for _, sc in c_otm.iterrows():  # short call
                        K3 = float(sc["strike_price"])
                        prem_sc = float(sc["premium"])
                        lc_cands = c_wings[c_wings["strike_price"].astype(float) > K3].sort_values("strike_price").head(4)
                        for _, lc in lc_cands.iterrows():
                            K4 = float(lc["strike_price"])
                            prem_lc = float(lc["premium"])

                            if not (K1 < K2 < K3 < K4):
                                continue

                            credit = (prem_sp - prem_lp) + (prem_sc - prem_lc)

                            payoff = credit                                      + (-max(K2 - S_target, 0.0) + max(K1 - S_target, 0.0))                                      + (-max(S_target - K3, 0.0) + max(S_target - K4, 0.0))

                            put_width = K2 - K1
                            call_width = K4 - K3

                            max_profit = credit
                            max_loss_low = credit - put_width
                            max_loss_high = credit - call_width
                            max_loss = min(max_loss_low, max_loss_high)

                            be_low = K2 - credit
                            be_high = K3 + credit
                            margin = max(put_width, call_width)
                            roi = (payoff / margin * 100.0) if margin > 0 else np.nan

                            legs = legs_html([
                                f"فروش Put: <b>{sp.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_sp:,.0f}')}",
                                f"خرید Put: <b>{lp.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_lp:,.0f}')}",
                                f"فروش Call: <b>{sc.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K3:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_sc:,.0f}')}",
                                f"خرید Call: <b>{lc.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K4:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem_lc:,.0f}')}",
                                f"اعتبار خالص (Credit) = {wrap_num_ltr(f'{credit:,.0f}')}",
                            ])

                            rows.append(make_row("Iron Condor (Short)", mat, legs, -credit, max_profit, max_loss,
                                                 f"{be_low:,.0f} , {be_high:,.0f}", payoff, roi, days_left))

        # --- Butterfly (Call): Buy 1 Call (K1) + Sell 2 Call (K2) + Buy 1 Call (K3)
        if len(calls_n) >= 3:
            c_sorted = calls_n.sort_values("strike_price")
            c_list = c_sorted.to_dict("records")[:14]  # cap
            for i in range(len(c_list) - 2):
                for j in range(i + 1, len(c_list) - 1):
                    for k in range(j + 1, len(c_list)):
                        c1, c2, c3 = c_list[i], c_list[j], c_list[k]
                        K1, K2, K3 = float(c1["strike_price"]), float(c2["strike_price"]), float(c3["strike_price"])
                        if not (K1 < K2 < K3):
                            continue
                        prem1, prem2, prem3 = float(c1["premium"]), float(c2["premium"]), float(c3["premium"])
                        debit = prem1 - 2.0 * prem2 + prem3
                        if debit <= 0:
                            continue

                        payoff = (max(S_target - K1, 0.0) - 2.0 * max(S_target - K2, 0.0) + max(S_target - K3, 0.0)) - debit
                        max_profit = (K2 - K1) - debit
                        max_loss = -debit
                        be_low = K1 + debit
                        be_high = K3 - debit
                        roi = (payoff / debit * 100.0) if debit > 0 else np.nan

                        legs = legs_html([
                            f"خرید Call: <b>{c1.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K1:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem1:,.0f}')}",
                            f"فروش 2× Call: <b>{c2.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K2:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem2:,.0f}')}",
                            f"خرید Call: <b>{c3.get('ticker')}</b> | قیمت اعمال={wrap_num_ltr(f'{K3:,.0f}')} | پرمیوم={wrap_num_ltr(f'{prem3:,.0f}')}",
                            f"هزینه خالص (Debit) = {wrap_num_ltr(f'{debit:,.0f}')}",
                        ])

                        rows.append(make_row("Long Call Butterfly", mat, legs, debit, max_profit, max_loss,
                                             f"{be_low:,.0f} , {be_high:,.0f}", payoff, roi, days_left))


    if not rows:
        st.warning("هیچ استراتژی قابل محاسبه‌ای پیدا نشد (ممکن است داده‌های Call/Put کافی نباشد).")
        st.stop()

    out = pd.DataFrame(rows)
    # فقط استراتژی‌های درخواستی
    wanted = {
        "Long Call","Long Put","Short Call","Short Put","Covered Call","Protective Put",
        "Bull Call Spread","Bear Call Spread","Bear Put Spread","Call Ratio Spread",
        "Long Straddle","Short Straddle","Long Strangle","Short Strangle",
        "Long Gut","Short Gut","Long Call Butterfly","Short Call Butterfly",
        "Long Box","Short Box","Collar","Conversion","Iron Condor (Short)"
    }
    out = out[out["استراتژی"].isin(wanted)].copy()


    # مرتب‌سازی کلی: بیشترین سود در S هدف
    out["سود در S هدف"] = pd.to_numeric(out["سود در S هدف"], errors="coerce")
    out_sorted = out.sort_values("سود در S هدف", ascending=False)

    best = out_sorted.iloc[0].to_dict()
    st.success(
        f"بهترین استراتژی بر اساس سود در S هدف: **{best['استراتژی']}** | سررسید **{best['سررسید']}** | سود: {best['سود در S هدف']:,.2f}"
    )
    st.markdown("**لِگ‌ها:**", unsafe_allow_html=True)
    st.markdown(safe_str(best.get("لِگ‌ها")), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 جدول بهترین‌ها (کلی)")
    show_cols = [
        "استراتژی",
        "سررسید",
        "مانده (روز)",
        "هزینه/دریافتی (پرمیوم خالص)",
        "حداکثر سود",
        "حداکثر زیان",
        "نقطه/نقاط سربه‌سر",
        "سود در S هدف",
        "ROI% تقریبی",
        "لِگ‌ها",
    ]

    # فرمت‌کردن برای نمایش RTL-safe
    view = out_sorted[show_cols].head(int(top_n)).copy()

    def _fmt(v, col):
        if v is None or _is_na(v):
            return "—"
        if col in ["استراتژی", "سررسید", "لِگ‌ها", "نقطه/نقاط سربه‌سر"]:
            return safe_str(v)
        if isinstance(v, str):
            return v
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return "—"
            if col in ["ROI% تقریبی"]:
                return wrap_num_ltr(f"{fv:,.2f}%")
            return wrap_num_ltr(f"{fv:,.2f}")
        except Exception:
            return safe_str(v)

    for c in view.columns:
        view[c] = view[c].apply(lambda x, cc=c: _fmt(x, cc))

    show_table(out_sorted[show_cols].copy(), "Top نتایج (قابل مرتب‌سازی + بدون نصفه شدن اعداد)", "tbl_top", int(top_n))

    st.markdown("---")
    
    st.markdown("---")
    st.markdown("### 📚 تب جداگانه برای هر استراتژی + راهنما")

    guides = {
        "Long Call": """**لانگ کال (Long Call)**  
- **ساختار:** خرید ۱ قرارداد Call  
- **دیدگاه:** صعودی (Bullish)  
- **حداکثر زیان:** پرمیوم پرداختی  
- **حداکثر سود:** نامحدود  
- **سربه‌سر:** قیمت اعمال + پرمیوم  
- **نکته:** برای حرکت‌های صعودی بزرگ مناسب است؛ زمان (Theta) به ضرر شماست.""",

        "Long Put": """**لانگ پوت (Long Put)**  
- **ساختار:** خرید ۱ قرارداد Put  
- **دیدگاه:** نزولی (Bearish)  
- **حداکثر زیان:** پرمیوم پرداختی  
- **حداکثر سود:** تقریباً تا (قیمت اعمال - پرمیوم) (در حد S→0)  
- **سربه‌سر:** قیمت اعمال − پرمیوم  
- **نکته:** برای ریزش‌های شدید یا پوشش ریسک مناسب است.""",

        "Short Call": """**شورت کال (Short Call)**  
- **ساختار:** فروش ۱ قرارداد Call  
- **دیدگاه:** خنثی تا نزولی  
- **حداکثر سود:** پرمیوم دریافتی  
- **حداکثر زیان:** نامحدود (با رشد قیمت پایه)  
- **سربه‌سر:** قیمت اعمال + پرمیوم  
- **نکته:** پرریسک؛ معمولاً با پوشش (Covered) یا اسپرد استفاده می‌شود.""",

        "Short Put": """**شورت پوت (Short Put)**  
- **ساختار:** فروش ۱ قرارداد Put (معمولاً Cash-Secured)  
- **دیدگاه:** خنثی تا صعودی  
- **حداکثر سود:** پرمیوم دریافتی  
- **حداکثر زیان:** محدود (تا S→0): پرمیوم − قیمت اعمال  
- **سربه‌سر:** قیمت اعمال − پرمیوم  
- **نکته:** شبیه سفارش خرید سهم با تخفیف (اگر Cash-Secured باشد).""",

        "Covered Call": """**کاورد کال (Covered Call)**  
- **ساختار:** خرید سهم پایه + فروش Call  
- **دیدگاه:** خنثی تا کمی صعودی  
- **حداکثر سود:** (قیمت اعمال − قیمت خرید سهم) + پرمیوم  
- **حداکثر زیان:** شبیه نگهداری سهم (با کمی کاهش به اندازه پرمیوم)  
- **سربه‌سر:** قیمت سهم − پرمیوم  
- **نکته:** مناسب برای درآمدزایی از سهم در بازار خنثی/آرام.""",

        "Protective Put": """**پروتکتیو پوت (Protective Put)**  
- **ساختار:** خرید سهم پایه + خرید Put  
- **دیدگاه:** صعودی با بیمه نزول  
- **حداکثر سود:** نامحدود  
- **حداکثر زیان:** محدود (تقریباً تا سطح K) + هزینه بیمه (پرمیوم)  
- **سربه‌سر:** قیمت سهم + پرمیوم  
- **نکته:** مثل «بیمه کردن» سهم است.""",

        "Bull Call Spread": """**بول کال اسپرد (Bull Call Spread)**  
- **ساختار:** خرید Call با K پایین + فروش Call با K بالاتر  
- **دیدگاه:** صعودی کنترل‌شده  
- **حداکثر زیان:** بدهی اولیه (Debit)  
- **حداکثر سود:** (اختلاف دو قیمت اعمال − Debit)  
- **سربه‌سر:** K پایین + Debit  
- **نکته:** هزینه کمتر از لانگ کال، اما سود سقف‌دار.""",

        "Bear Call Spread": """**بیر کال اسپرد (Bear Call Spread)**  
- **ساختار:** فروش Call با K پایین + خرید Call با K بالاتر (Credit Spread)  
- **دیدگاه:** خنثی تا نزولی  
- **حداکثر سود:** اعتبار دریافتی (Credit)  
- **حداکثر زیان:** (اختلاف قیمت اعمال − Credit)  
- **سربه‌سر:** K پایین + Credit  
- **نکته:** برای بازارهای رنج/نزولی ملایم.""",

        "Bear Put Spread": """**بیر پوت اسپرد (Bear Put Spread)**  
- **ساختار:** خرید Put با K بالاتر + فروش Put با K پایین‌تر  
- **دیدگاه:** نزولی کنترل‌شده  
- **حداکثر زیان:** Debit  
- **حداکثر سود:** (اختلاف قیمت اعمال − Debit)  
- **سربه‌سر:** K بالاتر − Debit  
- **نکته:** هزینه کمتر از لانگ پوت، اما سود سقف‌دار.""",

        "Call Ratio Spread": """**کال ریشیو اسپرد (Call Ratio Spread)**  
- **ساختار رایج (1×2):** خرید ۱ Call با K پایین + فروش ۲ Call با K بالاتر  
- **دیدگاه:** صعودی تا محدوده‌ای (اما با ریسک رشد شدید)  
- **حداکثر سود:** معمولاً در حوالی K بالاتر رخ می‌دهد  
- **حداکثر زیان:** می‌تواند نامحدود شود (اگر تعداد شورت بیشتر باشد)  
- **نکته:** استراتژی حرفه‌ای؛ مدیریت ریسک و وجه تضمین مهم است.""",

        "Long Straddle": """**لانگ استردل (Long Straddle)**  
- **ساختار:** خرید Call + خرید Put با یک قیمت اعمال (نزدیک ATM)  
- **دیدگاه:** انتظار نوسان شدید (بی‌جهت)  
- **حداکثر زیان:** مجموع پرمیوم‌ها  
- **حداکثر سود:** نامحدود  
- **سربه‌سر:** K ± مجموع پرمیوم  
- **نکته:** مناسب قبل از خبر/رویداد مهم.""",

        "Short Straddle": """**شورت استردل (Short Straddle)**  
- **ساختار:** فروش Call + فروش Put با یک قیمت اعمال  
- **دیدگاه:** انتظار بازار رنج و کاهش نوسان  
- **حداکثر سود:** مجموع پرمیوم دریافتی  
- **حداکثر زیان:** نامحدود  
- **سربه‌سر:** K ± مجموع پرمیوم  
- **نکته:** پرریسک و نیازمند مدیریت وجه تضمین.""",

        "Long Strangle": """**لانگ استرنگل (Long Strangle)**  
- **ساختار:** خرید Put با K پایین‌تر + خرید Call با K بالاتر  
- **دیدگاه:** انتظار نوسان؛ هزینه کمتر از استردل  
- **حداکثر زیان:** مجموع پرمیوم‌ها  
- **حداکثر سود:** نامحدود  
- **سربه‌سر:** Kp − net و Kc + net  
- **نکته:** برای حرکت بزرگ در هر جهت.""",

        "Short Strangle": """**شورت استرنگل (Short Strangle)**  
- **ساختار:** فروش Put (OTM) + فروش Call (OTM)  
- **دیدگاه:** بازار رنج  
- **حداکثر سود:** مجموع پرمیوم دریافتی  
- **حداکثر زیان:** نامحدود  
- **سربه‌سر:** Kp − credit و Kc + credit  
- **نکته:** نسبت به شورت استردل کمی امن‌تر اما همچنان پرریسک.""",

        "Long Gut": """**لانگ گات (Long Gut)**  
- **ساختار:** خرید Call (ITM) + خرید Put (ITM) با دو قیمت اعمال متفاوت  
- **دیدگاه:** انتظار حرکت بزرگ در هر جهت (با هزینه/ساختار متفاوت از استردل)  
- **حداکثر زیان:** مجموع پرمیوم‌ها  
- **حداکثر سود:** نامحدود  
- **نکته:** کمتر رایج؛ قراردادهای ITM ممکن است نقدشوندگی متفاوتی داشته باشند.""",

        "Short Gut": """**شورت گات (Short Gut)**  
- **ساختار:** فروش Call (ITM) + فروش Put (ITM)  
- **دیدگاه:** بازار رنج با دریافت پرمیوم بیشتر (اما ریسک بالا)  
- **حداکثر سود:** مجموع پرمیوم دریافتی  
- **حداکثر زیان:** نامحدود  
- **نکته:** حرفه‌ای/پرریسک.""",

        "Long Call Butterfly": """**لانگ کال باترفلای (Long Call Butterfly)**  
- **ساختار:** خرید Call(K1) + فروش 2× Call(K2) + خرید Call(K3) (K1<K2<K3)  
- **دیدگاه:** انتظار بازار نزدیک K2 در سررسید  
- **حداکثر زیان:** Debit  
- **حداکثر سود:** محدود (معمولاً حوالی K2)  
- **نکته:** برای کاهش هزینه و شرط‌بندی روی نوسان پایین.""",

        "Short Call Butterfly": """**شورت کال باترفلای (Short Call Butterfly)**  
- **ساختار:** معکوس باترفلای لانگ (دریافت Credit)  
- **دیدگاه:** انتظار حرکت دور از K2  
- **حداکثر سود:** Credit  
- **حداکثر زیان:** محدود  
- **نکته:** ریسک/بازده محدود، اما نیازمند مدیریت.""",

        "Long Box": """**لانگ باکس (Long Box)**  
- **ساختار:** (Bull Call Spread) + (Bear Put Spread) با دو استرایک مشترک  
- **نتیجه:** پرداخت ثابت (K2−K1) در سررسید (شبیه وام دادن)  
- **نکته:** اختلاف قیمت با ارزش تئوریک = نرخ بهره/هزینه‌ها.""",

        "Short Box": """**شورت باکس (Short Box)**  
- **ساختار:** معکوس لانگ باکس (دریافت ثابت در ابتدا، پرداخت (K2−K1) در سررسید)  
- **نتیجه:** شبیه وام گرفتن  
- **نکته:** هزینه‌ها/وجه تضمین مهم است.""",

        "Collar": """**کالر (Collar)**  
- **ساختار:** خرید سهم + خرید Put (OTM) + فروش Call (OTM)  
- **دیدگاه:** محدود کردن زیان و سقف‌دار کردن سود  
- **حداکثر سود:** محدود (تا Kc)  
- **حداکثر زیان:** محدود (تا Kp)  
- **نکته:** مناسب مدیریت ریسک روی سهم.""",

        "Conversion": """**کانورژن (Conversion)**  
- **ساختار:** خرید سهم + خرید Put + فروش Call (هم‌استرایک)  
- **نتیجه:** سود/زیان تقریباً ثابت (آربیتراژی نظری)  
- **نکته:** در عمل کارمزد/اسپرد/محدودیت‌ها تعیین‌کننده‌اند.""",
 "Iron Condor (Short)": """**آیرون کندور (Iron Condor - Short/Credit)**  
+- **ساختار:** فروش Put (OTM) + خرید Put پایین‌تر (Wing) + فروش Call (OTM) + خرید Call بالاتر (Wing)  
+- **دیدگاه:** بازار رنج / کاهش نوسان (Short Vol)  
+- **حداکثر سود:** اعتبار (Credit) دریافتی  
+- **حداکثر زیان:** محدود (عرض یکی از بال‌ها − Credit)  
+- **سربه‌سر:** K_put_short − Credit  و  K_call_short + Credit  
+- **نکته:** نسبت به Short Strangle ریسک محدودتر است و برای بازارهای آرام مناسب است.""",
    }

    strat_order = [
        "Long Call","Long Put","Short Call","Short Put","Covered Call",
        "Protective Put","Bull Call Spread","Bear Call Spread","Bear Put Spread",
        "Call Ratio Spread","Long Straddle","Short Straddle","Long Strangle","Short Strangle",
        "Long Gut","Short Gut","Long Call Butterfly","Short Call Butterfly","Long Box","Short Box",
        "Collar","Conversion","Iron Condor (Short)"
    ]

    tabs = st.tabs([f"📌 {n}" for n in strat_order])
    for t, strat_name in zip(tabs, strat_order):
        with t:
            render_rtl_guide(guides.get(strat_name, ""))
            sub = out[out["استراتژی"] == strat_name].copy()
            if len(sub) == 0:
                st.info("برای این استراتژی ترکیب قابل محاسبه‌ای در داده‌های فعلی پیدا نشد.")
                continue
            sub = sub.sort_values("سود در S هدف", ascending=False).head(int(top_n))
            show_table(sub[show_cols].copy(), f"نتایج: {strat_name}", f"tbl_{strat_name}", int(top_n))
# ============================================================
# TAB 3: History (Option + Underlying) like before
# ============================================================
with tab_history:
    st.subheader("📉 تاریخچه واقعی قرارداد + سهم پایه")

    row = st.session_state.get("selected_row")
    if not row:
        st.info("اول در تب فیلترها یک قرارداد انتخاب کن.")
        st.stop()

    opt_ticker = safe_str(row.get("ticker"))
    st.markdown(f"**قرارداد انتخاب‌شده:** {opt_ticker}")
    st.markdown(f"**سهم پایه:** {ua_ticker}")

    st.markdown(
        "<div class='small'>برای تاریخچه واقعی از finpy-tse استفاده می‌کنیم. اگر نصب نیست: <code>pip install finpy-tse</code></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        start_j = st.text_input("از تاریخ (جلالی) - مثال 1402-01-01", value="1402-01-01", key="h_start")
    with c2:
        end_j = st.text_input("تا تاریخ (جلالی)", value=jdatetime.date.today().strftime("%Y-%m-%d"), key="h_end")
    with c3:
        price_field = st.selectbox("ستون قیمت", ["Close", "Final", "Adj Close"], index=0, key="h_pf")

    if st.button("📥 دریافت تاریخچه و رسم نمودارها", key="h_btn"):
        try:
            import finpy_tse as fpy
        except Exception:
            st.error("پکیج finpy-tse نصب نیست. نصب کن:  pip install finpy-tse")
            st.stop()

        def get_hist(symbol: str):
            return fpy.Get_Price_History(
                stock=symbol,
                start_date=start_j,
                end_date=end_j,
                ignore_date=False,
                adjust_price=False,
                show_weekday=False,
                double_date=False,
            )

        def normalize(df: pd.DataFrame):
            if df is None or len(df) == 0:
                return None
            d = df.copy().reset_index()
            date_col = None
            for c in d.columns:
                if "date" in str(c).lower() or "تاریخ" in str(c):
                    date_col = c
                    break
            if date_col is None:
                date_col = d.columns[0]
            d.rename(columns={date_col: "Date"}, inplace=True)
            try:
                d["Date"] = pd.to_datetime(d["Date"])
            except Exception:
                pass
            return d

        def pick_price_col(df: pd.DataFrame):
            if df is None:
                return None
            if price_field == "Close":
                cand = ["Close", "close", "PClosing", "قیمت پایانی", "پایانی"]
            elif price_field == "Final":
                cand = ["Final", "final", "PDrCotVal", "آخرین", "قیمت آخرین"]
            else:
                cand = ["Adj Close", "AdjClose", "adjclose", "قیمت تعدیل", "تعدیل"]
            for c in cand:
                if c in df.columns:
                    return c
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            return num_cols[0] if num_cols else None
        def get_ohlc_cols(df: pd.DataFrame):
            if df is None or len(df) == 0:
                return None

            open_cand = ["Open", "open", "POpen", "بازگشایی", "باز", "قیمت بازگشایی"]
            high_cand = ["High", "high", "PHigh", "بیشترین", "سقف", "قیمت بیشینه"]
            low_cand  = ["Low", "low", "PLow", "کمترین", "کف", "قیمت کمینه"]
            close_cand = ["Close", "close", "PClosing", "پایانی", "قیمت پایانی", "Adj Close", "AdjClose", "پایانی تعدیل‌شده"]

            def pick(cands):
                for c in cands:
                    if c in df.columns:
                        return c
                return None

            o = pick(open_cand)
            h = pick(high_cand)
            l = pick(low_cand)
            c = pick(close_cand)

            if any(x is None for x in [o, h, l, c]):
                num_cols = [cc for cc in df.columns if cc != "Date" and pd.api.types.is_numeric_dtype(df[cc])]
                if len(num_cols) >= 4:
                    o = o or num_cols[0]
                    h = h or num_cols[1]
                    l = l or num_cols[2]
                    c = c or num_cols[3]

            if all(x is not None for x in [o, h, l, c]):
                return {"open": o, "high": h, "low": l, "close": c}
            return None

        def plot_candlestick(df: pd.DataFrame, title: str):
            if df is None or len(df) == 0:
                st.warning("داده‌ای برای کندلی موجود نیست.")
                return
            cols = get_ohlc_cols(df)
            if not cols:
                st.warning("ستون‌های OHLC برای رسم کندلی پیدا نشد.")
                return
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["Date"],
                        open=df[cols["open"]],
                        high=df[cols["high"]],
                        low=df[cols["low"]],
                        close=df[cols["close"]],
                    )
                ]
            )
            fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)


        opt_hist = None
        ua_hist = None
        try:
            opt_hist = normalize(get_hist(opt_ticker))
        except Exception as e:
            st.warning(f"تاریخچه آپشن دریافت نشد: {e}")

        try:
            ua_hist = normalize(get_hist(ua_ticker))
        except Exception as e:
            st.warning(f"تاریخچه سهم پایه دریافت نشد: {e}")

        left, right = st.columns(2)

        with right:
            st.markdown("### نمودار آپشن")
            if opt_hist is None:
                st.warning("داده تاریخچه آپشن موجود نیست.")
            else:
                pcol = pick_price_col(opt_hist)
                if not pcol:
                    st.warning("ستون قیمت برای آپشن پیدا نشد.")
                else:
                    fig = px.line(opt_hist, x="Date", y=pcol, title=f"تاریخچه قیمت آپشن: {opt_ticker}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### کندلی")
                    plot_candlestick(opt_hist, title=f"کندلی آپشن: {opt_ticker}")

        with left:
            st.markdown("### نمودار سهم پایه")
            if ua_hist is None:
                st.warning("داده تاریخچه سهم پایه موجود نیست.")
            else:
                pcol = pick_price_col(ua_hist)
                if not pcol:
                    st.warning("ستون قیمت برای سهم پایه پیدا نشد.")
                else:
                    fig = px.line(ua_hist, x="Date", y=pcol, title=f"تاریخچه قیمت سهم پایه: {ua_ticker}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### کندلی")
                    plot_candlestick(ua_hist, title=f"کندلی سهم پایه: {ua_ticker}")

        st.markdown("---")

        def fa_hist_cols(df: pd.DataFrame):
            # تبدیل اسامی انگلیسی رایج تاریخچه به فارسی
            if df is None or len(df) == 0:
                return df
            ren = {
                "Date": "تاریخ",
                "Open": "بازگشایی",
                "High": "بیشترین",
                "Low": "کمترین",
                "Close": "پایانی",
                "Final": "آخرین",
                "Adj Close": "پایانی تعدیل‌شده",
                "Volume": "حجم",
                "Value": "ارزش",
                "No": "تعداد",
            }
            return df.rename(columns={c: ren.get(c, c) for c in df.columns})

        if opt_hist is not None:
            pcol = pick_price_col(opt_hist)
            show_cols = ["Date"] + ([pcol] if pcol else [])
            preview = opt_hist[show_cols].tail(200).copy() if all(c in opt_hist.columns for c in show_cols) else opt_hist.tail(200).copy()
            preview = fa_hist_cols(preview)
            for ccol in preview.columns:
                if ccol != "تاریخ" and pd.api.types.is_numeric_dtype(preview[ccol]):
                    preview[ccol] = preview[ccol].apply(lambda x: wrap_num_ltr(fmt_num(x, 4)))
            render_table_html(preview, height_px=420, title="جدول تاریخچه آپشن (۲۰۰ ردیف آخر)")

        if ua_hist is not None:
            pcol = pick_price_col(ua_hist)
            show_cols = ["Date"] + ([pcol] if pcol else [])
            preview = ua_hist[show_cols].tail(200).copy() if all(c in ua_hist.columns for c in show_cols) else ua_hist.tail(200).copy()
            preview = fa_hist_cols(preview)
            for ccol in preview.columns:
                if ccol != "تاریخ" and pd.api.types.is_numeric_dtype(preview[ccol]):
                    preview[ccol] = preview[ccol].apply(lambda x: wrap_num_ltr(fmt_num(x, 4)))
            render_table_html(preview, height_px=420, title="جدول تاریخچه سهم پایه (۲۰۰ ردیف آخر)")

    st.markdown("---")
    st.markdown("### (اختیاری) LOB تاریخی آپشن")

    c1, c2 = st.columns(2)
    with c1:
        jalali_lob = st.text_input("تاریخ جلالی برای LOB (مثلاً 1403-10-24)", value="", key="lob_date")
    with c2:
        show_n = st.number_input("تعداد ردیف برای نمایش", 50, 2000, 250, 50, key="lob_n")

    if st.button("دریافت LOB و رسم Bid/Ask", key="lob_btn"):
        tse_code = safe_str(row.get("tse_code"))
        if tse_code == "—" or not jalali_lob.strip():
            st.warning("کد TSE یا تاریخ جلالی معتبر نیست.")
        else:
            try:
                lob = fetch_historical_lob(tse_code=tse_code, jalali_date=jalali_lob.strip())
                if lob is None or len(lob) == 0:
                    st.warning("برای این تاریخ، داده LOB موجود نیست.")
                else:
                    # فارسی‌سازی سرستون‌های معروف LOB
                    lob = lob.rename(columns={
                        "bid_price": "قیمت خرید",
                        "bid_volume": "حجم خرید",
                        "ask_price": "قیمت فروش",
                        "ask_volume": "حجم فروش",
                        "time": "زمان",
                        "timestamp": "زمان",
                    })
                    render_table_html(lob.tail(int(show_n)), height_px=420, title="نمونه LOB")

                    time_col = next((c for c in ["زمان", "Time", "timestamp", "t"] if c in lob.columns), None)
                    bid_col = next((c for c in lob.columns if str(c) in ["قیمت خرید", "bid_price"]), None)
                    ask_col = next((c for c in lob.columns if str(c) in ["قیمت فروش", "ask_price"]), None)

                    y_cols = [c for c in [bid_col, ask_col] if c]
                    if time_col and y_cols:
                        fig = px.line(lob, x=time_col, y=y_cols, title=f"Bid/Ask — {opt_ticker} — {jalali_lob}")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"خطا در دریافت/نمایش LOB: {e}")