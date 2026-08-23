"""
XAUUSD signal scanner.
Runs on a schedule (GitHub Actions). Pulls live gold price/tick data from the free
xaus.com API, builds 5m/15m/30m/1H candles, checks the full rule set, and sends a
Telegram notification when a BUY/SELL/PRE-SIGNAL fires. State (last signal, to avoid
duplicate alerts) is persisted in state.json and committed back to the repo.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta

import requests

# ============================= CONFIG =============================
SYMBOL = "xau"
INTRADAY_HOURS = 48
ATR_PERIOD = 15
RSI_PERIOD = 14
FAST_MA_PROXIMITY_USD = 5
MIN_BREAK_USD = 5
LINE_DEVIATION_USD = 3          # trendline touch tolerance (for line validity itself)
TRENDLINE_RETEST_DEVIATION_USD = 2  # looser tolerance just for confirming a retest touch
SR_DEVIATION_USD = 5            # support/resistance touch tolerance AND retest tolerance
SWING_LOOKBACK = 5
RISK_REWARD = 1
PRE_SIGNAL_MIN_CONDITIONS = 4   # fire a pre-signal once at least this many of the 6 CORE conditions are met
FULL_SIGNAL_MIN_CONDITIONS = 6  # full signal needs all 6 core, OR 5/6 core + at least 1 confluence bonus
# a trendline must be built from swing touches within this many recent candles per timeframe
TRENDLINE_RECENCY_CANDLES = {5: 150, 15: 190, 30: 96, 60: 48}
MIN_TRENDLINE_TOUCHES = 3            # minimum gate; bounce magnitude is the real differentiator
MIN_TRENDLINE_SPAN_CANDLES = 15      # touches must be spread out, not bunched together
MAX_TRENDLINE_SLOPE_USD_PER_CANDLE = 3.0  # reject near-vertical lines (unsustainable moves)
# --- 5m entry-line liveness (ATR-scaled so it adapts to volatility) ---
LINE_MAX_DISTANCE_ATR = 8.0          # line must project within 8x ATR of current price
LINE_ACTIVITY_WINDOW_CANDLES = 100   # last touch/break must be within this many candles
LINE_SPENT_ATR = 4.0                 # if price ever went >4x ATR beyond the line, it's spent forever
# --- bounce scoring: how many candles after a touch to measure the reaction ---
BOUNCE_MEASURE_CANDLES = 6
MAX_LINE_DISTANCE_USD = 30  # a trendline projecting further than this from current price is treated as stale/unreliable
# session windows in PKT: (start_hour,start_min,end_hour,end_min)
SESSION_WINDOWS = [(7, 0, 16, 0), (20, 0, 23, 0)]

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
# --- Data sanity: these WARN, they never block a trade. Deliberately loose, because a
# fast move is the signal in a break-and-retest strategy, not a data error. ---
DATA_STALE_WARN_MINUTES = 15      # newest tick older than this -> flag as stale
DATA_SPOT_MISMATCH_WARN = 5.0     # spot vs latest candle disagree by more than this
DATA_JUMP_WARN_PCT = 3.0          # price moved this % since last scan
DATA_WARN_COOLDOWN_MINUTES = 60   # don't repeat the same warning more often than this
# --- Daily heartbeat ---
HEARTBEAT_HOUR_PKT = 23           # send the daily summary at this PKT hour
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ============================= DATA FETCH =============================
def fetch_intraday():
    url = f"https://xaus.com/api/v1/intraday?symbol={SYMBOL}&hours={INTRADAY_HOURS}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json().get("points", [])


def fetch_spot():
    url = "https://xaus.com/api/v1/spot?compact=1"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


# ============================= CANDLES =============================
def ts_of(point):
    t = point.get("t")
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000
    if isinstance(t, (int, float)):
        return t if t > 2e10 else t * 1000
    return datetime.now(timezone.utc).timestamp() * 1000


def build_candles(points, interval_min):
    """Build candles on FIXED CLOCK INTERVALS.

    The free price feed delivers ticks irregularly - sometimes several minutes apart.
    The previous version created one candle per tick received, so when ticks were sparse
    a '5m candle' could actually span 10+ real minutes. That made every indicator wrong:
    a '21-period MA on 5m' was really averaging a much longer, older stretch of price than
    MT5's, so it could sit far from current price while the bot still judged it 'close'.

    Now every interval gets exactly one candle. Intervals with no ticks carry the previous
    close forward (flat candle), which is what a real chart shows during quiet periods.
    """
    if not points:
        return []
    pts = sorted(points, key=ts_of)
    bucket_ms = interval_min * 60000

    # Group ticks by their clock bucket
    buckets = {}
    for p in pts:
        price = p.get("p")
        if price is None:
            continue
        bs = math.floor(ts_of(p) / bucket_ms) * bucket_ms
        if bs not in buckets:
            buckets[bs] = {"o": price, "h": price, "l": price, "c": price}
        else:
            b = buckets[bs]
            b["h"] = max(b["h"], price)
            b["l"] = min(b["l"], price)
            b["c"] = price

    if not buckets:
        return []

    first_bs, last_bs = min(buckets), max(buckets)
    out = []
    prev_close = None
    bs = first_bs
    while bs <= last_bs:
        if bs in buckets:
            b = buckets[bs]
            out.append({"t": bs, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]})
            prev_close = b["c"]
        elif prev_close is not None:
            # No ticks this interval - carry the last close forward as a flat candle,
            # so the candle count stays true to elapsed clock time.
            out.append({"t": bs, "o": prev_close, "h": prev_close,
                        "l": prev_close, "c": prev_close})
        bs += bucket_ms
    return out


# ============================= INDICATORS =============================
def smma(vals, period):
    """Smoothed Moving Average - matches MT5's MA method 'Smoothed' (SMMA/Wilder's).

    MT5 seeds the first value with a simple average of the first `period` values, then:
        SMMA[i] = (SMMA[i-1] * (period - 1) + value[i]) / period

    This is NOT the same as a Simple MA: old data is never fully discarded, so SMMA
    reacts more slowly and sits further from price in a trend. Using SMA here was making
    the bot's 21MA disagree with the chart.
    """
    if len(vals) < period:
        return None
    val = sum(vals[:period]) / period  # seed
    for v in vals[period:]:
        val = (val * (period - 1) + v) / period
    return val


def sma(vals, period):
    if len(vals) < period:
        return None
    return sum(vals[-period:]) / period


def rsi_series(closes, period):
    out = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g, ls = max(d, 0), max(-d, 0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + ls) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    return out


def atr(candles, period):
    """Wilder's smoothed ATR - matches MT5's built-in ATR indicator (not a plain average)."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        tr = max(cur["h"] - cur["l"], abs(cur["h"] - prev["c"]), abs(cur["l"] - prev["c"]))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr_val = sum(trs[:period]) / period  # seed value: simple average of first `period` TRs
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period  # Wilder's running smoothing
    return atr_val


# ============================= SWINGS =============================
def find_swings(candles, lookback):
    highs, lows = [], []
    for i in range(lookback, len(candles) - lookback):
        c = candles[i]
        is_high = all(candles[j]["h"] < c["h"] for j in range(i - lookback, i + lookback + 1) if j != i)
        is_low = all(candles[j]["l"] > c["l"] for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_high:
            highs.append({"i": i, "price": c["h"]})
        if is_low:
            lows.append({"i": i, "price": c["l"]})
    return highs, lows


# ============================= TRENDLINES (fit-and-score) =============================


def measure_bounce(candles, touch_idx, line_slope, line_intercept, side, lookahead=BOUNCE_MEASURE_CANDLES):
    """How far did price travel AWAY from the line in the candles right after a touch?

    A big bounce means the level had real defenders. For a break-and-retest trader that's
    exactly what you want: the more energy a level absorbed while holding, the more gets
    released (trapped orders, stop runs) when it finally breaks.
    """
    end = min(len(candles), touch_idx + lookahead + 1)
    best = 0.0
    for i in range(touch_idx + 1, end):
        lv = line_slope * i + line_intercept
        c = candles[i]
        if side == "resistance":
            dist = lv - c["l"]      # resistance holds -> price pushed DOWN away from line
        else:
            dist = c["h"] - lv      # support holds -> price pushed UP away from line
        if dist > best:
            best = dist
    return best


def line_is_live(candles, slope, intercept, side, touch_idxs, price, atr_val):
    """Liveness checks for the 5m ENTRY line (not applied to obstacle lines).

    1. Line must project within LINE_MAX_DISTANCE_ATR x ATR of current price.
    2. Last touch/break must be within LINE_ACTIVITY_WINDOW_CANDLES.
    3. Not spent: if price EVER travelled more than LINE_SPENT_ATR x ATR beyond the line
       (on the far side only - drifting away on the SAME side never counts as spent),
       the structure has delivered its move and is discarded permanently.
    """
    if not atr_val or atr_val <= 0:
        return False, "no ATR"
    n = len(candles)

    line_now = slope * (n - 1) + intercept
    if abs(line_now - price) > LINE_MAX_DISTANCE_ATR * atr_val:
        return False, "line too far from price"

    if touch_idxs and (n - 1 - max(touch_idxs)) > LINE_ACTIVITY_WINDOW_CANDLES:
        return False, "line inactive too long"

    spent_threshold = LINE_SPENT_ATR * atr_val
    for i in range(n):
        lv = slope * i + intercept
        c = candles[i]
        if side == "resistance":
            # spent only if price broke ABOVE and ran far beyond
            if c["c"] - lv > spent_threshold:
                return False, "line already spent (price ran far above)"
        else:
            # spent only if price broke BELOW and ran far beyond
            if lv - c["c"] > spent_threshold:
                return False, "line already spent (price ran far below)"

    return True, None


def detect_trendline(candles, side, window, deviation=LINE_DEVIATION_USD,
                     min_touches=MIN_TRENDLINE_TOUCHES, min_span=MIN_TRENDLINE_SPAN_CANDLES,
                     price=None, atr_val=None, apply_liveness=False):
    """Fit-and-score trendline detection.

    Draws a candidate line through EVERY pair of candle wicks in the window, then counts
    how many wicks come within `deviation` of that line. This avoids the old bug where
    genuine touches on a sloped line disqualified each other.

    side='resistance' -> uses candle highs, line sits above price (declining line in a fall)
    side='support'    -> uses candle lows,  line sits below price (inclining line in a rise)

    SCORING (tuned for a break-and-retest trader):
      * PRIMARY: total bounce magnitude, ATR-normalised. A level that produced big
        reactions had real defenders; when it breaks, those trapped positions fuel the move.
        This is what determines whether a break can actually run 2x ATR to target.
      * GATE: minimum touch count and minimum time span (structure must be real).
      * PENALTY: candle closes past the line before the break (line not respected).
      * FILTER: slope sanity, plus optional liveness checks for the 5m entry line.
    """
    n = len(candles)
    if n < min_span + 2:
        return None
    start = max(0, n - window)
    idxs = list(range(start, n))
    if len(idxs) < 3:
        return None

    key = "h" if side == "resistance" else "l"
    pts = [(i, candles[i][key]) for i in idxs]

    best = None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            i1, p1 = pts[a]
            i2, p2 = pts[b]
            if i2 == i1 or (i2 - i1) < min_span:
                continue

            slope = (p2 - p1) / (i2 - i1)
            intercept = p1 - slope * i1

            if abs(slope) > MAX_TRENDLINE_SLOPE_USD_PER_CANDLE:
                continue

            touch_idxs = [i for i, pval in pts
                          if abs(pval - (slope * i + intercept)) <= deviation]
            if len(touch_idxs) < min_touches:
                continue

            span = max(touch_idxs) - min(touch_idxs)
            if span < min_span:
                continue

            if apply_liveness:
                ok, _reason = line_is_live(candles, slope, intercept, side,
                                            touch_idxs, price, atr_val)
                if not ok:
                    continue

            # --- cleanliness: candle closes past the line before the last touch ---
            violations = 0
            last_touch = max(touch_idxs)
            for i in idxs:
                if i > last_touch:
                    break
                lv = slope * i + intercept
                c = candles[i]["c"]
                if side == "resistance" and c > lv + deviation:
                    violations += 1
                elif side == "support" and c < lv - deviation:
                    violations += 1

            # --- PRIMARY METRIC: bounce magnitude across all touches ---
            bounces = [measure_bounce(candles, ti, slope, intercept, side) for ti in touch_idxs]
            total_bounce = sum(bounces)
            avg_bounce = total_bounce / len(bounces) if bounces else 0.0
            # normalise by ATR so this means the same thing in quiet and volatile conditions
            norm = atr_val if (atr_val and atr_val > 0) else 1.0
            bounce_score = (total_bounce / norm) * 6.0 + (avg_bounce / norm) * 10.0

            score = bounce_score + (len(touch_idxs) * 2) + (span * 0.2) - (violations * 8)

            if best is None or score > best["score"]:
                best = {
                    "slope": slope, "intercept": intercept,
                    "touches": len(touch_idxs), "span": span,
                    "violations": violations, "score": score,
                    "total_bounce": total_bounce, "avg_bounce": avg_bounce,
                    "last_touch_i": last_touch, "side": side,
                }
    return best


def detect_nearest_obstacle_line(candles, side, window, entry, target,
                                 deviation=LINE_DEVIATION_USD,
                                 min_touches=MIN_TRENDLINE_TOUCHES,
                                 min_span=MIN_TRENDLINE_SPAN_CANDLES):
    """For higher-timeframe OBSTACLE lines (15m/30m/1H).

    Unlike the 5m entry line, freshness doesn't matter here - a dormant wall still stops
    price when it gets there. So no liveness filtering. What matters is which qualifying
    line sits NEAREST to the entry->target path.
    """
    n = len(candles)
    if n < min_span + 2:
        return None
    start = max(0, n - window)
    idxs = list(range(start, n))
    if len(idxs) < 3:
        return None

    key = "h" if side == "resistance" else "l"
    pts = [(i, candles[i][key]) for i in idxs]
    lo, hi = (entry, target) if target > entry else (target, entry)

    best = None
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            i1, p1 = pts[a]
            i2, p2 = pts[b]
            if i2 == i1 or (i2 - i1) < min_span:
                continue
            slope = (p2 - p1) / (i2 - i1)
            intercept = p1 - slope * i1
            if abs(slope) > MAX_TRENDLINE_SLOPE_USD_PER_CANDLE:
                continue
            touch_idxs = [i for i, pval in pts
                          if abs(pval - (slope * i + intercept)) <= deviation]
            if len(touch_idxs) < min_touches:
                continue
            if (max(touch_idxs) - min(touch_idxs)) < min_span:
                continue

            line_now = slope * (n - 1) + intercept
            if not (lo <= line_now <= hi):
                continue  # not in the path, not an obstacle

            dist_from_entry = abs(line_now - entry)
            if best is None or dist_from_entry < best["dist_from_entry"]:
                best = {"slope": slope, "intercept": intercept, "side": side,
                        "touches": len(touch_idxs), "line_now": line_now,
                        "dist_from_entry": dist_from_entry}
    return best


def line_value_at(line, idx):
    return line["slope"] * idx + line["intercept"]




def retest_touch_found(candles, level_at, deviation, lookback=10, return_detail=False):
    """Check the recent candles (excluding the very latest, which is the break/rejection
    candle itself) for a genuine touch back to the line/level — i.e. price actually came
    back and interacted with it after breaking, not just currently sitting past it.

    With return_detail=True returns (found, closest_gap, level_at_closest) so the message
    can report exactly how close price came, instead of only asserting that it touched.
    """
    start = max(0, len(candles) - lookback)
    end = len(candles) - 1  # exclude the current/last candle
    found = False
    closest = None
    closest_level = None
    for i in range(start, end):
        level = level_at(i)
        if level is None:
            continue
        c = candles[i]
        if (c["l"] - deviation) <= level <= (c["h"] + deviation):
            found = True
            gap = 0.0  # the candle's range actually contains the level
        elif level > c["h"]:
            gap = level - c["h"]
        else:
            gap = c["l"] - level
        if closest is None or gap < closest:
            closest, closest_level = gap, level
    if return_detail:
        return found, closest, closest_level
    return found


# ============================= SUPPORT / RESISTANCE (1h) =============================
def detect_sr(candles, deviation):
    highs, lows = find_swings(candles, SWING_LOOKBACK)

    def cluster(points):
        clusters = []
        for p in points:
            found = next((c for c in clusters if abs(c["avg"] - p["price"]) <= deviation), None)
            if found:
                found["pts"].append(p)
                found["avg"] = sum(x["price"] for x in found["pts"]) / len(found["pts"])
            else:
                clusters.append({"avg": p["price"], "pts": [p]})
        return [c for c in clusters if len(c["pts"]) >= 3]

    return cluster(highs), cluster(lows)


# ============================= LINE STRIKE =============================
def detect_line_strike(candles):
    """3-line strike: 3 candles one way, then a 4th opposite candle that swallows them.

    Returns (kind, reversal_direction, detail) where detail carries the real measurements
    behind the call - the rejection candle's open/close, its body size, and the level it
    had to close beyond to qualify - so the message can show evidence, not just a verdict.
    """
    if len(candles) < 4:
        return None, None, None
    c1, c2, c3, c4 = candles[-4:]
    direction = "up" if c1["c"] > c1["o"] else "down"
    same3 = all((c["c"] > c["o"]) if direction == "up" else (c["c"] < c["o"]) for c in (c1, c2, c3))
    if not same3:
        return None, None, None
    c4dir = "up" if c4["c"] > c4["o"] else "down"
    if c4dir == direction:
        return None, None, None

    range_low, range_high = min(c1["l"], c2["l"], c3["l"]), max(c1["h"], c2["h"], c3["h"])
    rev = "down" if direction == "up" else "up"
    body = abs(c4["c"] - c4["o"])

    if (direction == "up" and c4["c"] < range_low) or (direction == "down" and c4["c"] > range_high):
        needed = range_low if direction == "up" else range_high
        return "3-line", rev, {"open": c4["o"], "close": c4["c"], "body": body,
                                "needed": needed, "swallowed": 3}
    if (direction == "up" and c4["c"] < c3["o"]) or (direction == "down" and c4["c"] > c3["o"]):
        return "2-line", rev, {"open": c4["o"], "close": c4["c"], "body": body,
                                "needed": c3["o"], "swallowed": 1}

    # Near miss: the 3-in-a-row and an opposing 4th candle are there, but it didn't close
    # far enough to swallow even one prior candle. Report how much short it fell.
    shortfall = (c4["c"] - c3["o"]) if direction == "up" else (c3["o"] - c4["c"])
    return None, rev, {"open": c4["o"], "close": c4["c"], "body": body,
                       "needed": c3["o"], "swallowed": 0,
                       "near_miss": True, "shortfall": abs(shortfall)}


# ============================= RSI DIVERGENCE =============================
def detect_divergence(candles, rsi_vals):
    """Detect regular/hidden RSI divergence.

    Also returns the exact swing prices and RSI readings that were compared, so the
    Telegram message can show the real evidence rather than just asserting a result.
    Shape: {"bullish": kind|None, "bearish": kind|None, "bullish_detail": {...}, ...}
    """
    highs, lows = find_swings(candles, 3)
    result = {"bullish": None, "bearish": None,
              "bullish_detail": None, "bearish_detail": None}

    def rsi_at(i):
        return rsi_vals[i] if i < len(rsi_vals) else None

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        r1, r2 = rsi_at(h1["i"]), rsi_at(h2["i"])
        if r1 is not None and r2 is not None:
            kind = None
            if h2["price"] > h1["price"] and r2 < r1:
                kind = "regular"
            elif h2["price"] < h1["price"] and r2 > r1:
                kind = "hidden"
            if kind:
                result["bearish"] = kind
                result["bearish_detail"] = {
                    "p1": h1["price"], "p2": h2["price"], "r1": r1, "r2": r2,
                }
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        r1, r2 = rsi_at(l1["i"]), rsi_at(l2["i"])
        if r1 is not None and r2 is not None:
            kind = None
            if l2["price"] < l1["price"] and r2 > r1:
                kind = "regular"
            elif l2["price"] > l1["price"] and r2 < r1:
                kind = "hidden"
            if kind:
                result["bullish"] = kind
                result["bullish_detail"] = {
                    "p1": l1["price"], "p2": l2["price"], "r1": r1, "r2": r2,
                }
    return result


# ============================= SESSION =============================
def toPKT_str(now_utc):
    """Current time rendered in PKT (UTC+5), for showing the real clock in messages."""
    return (now_utc + timedelta(hours=5)).strftime("%H:%M")


def in_session_window(now_utc):
    pkt = now_utc + timedelta(hours=5)
    mins = pkt.hour * 60 + pkt.minute
    for h1, m1, h2, m2 in SESSION_WINDOWS:
        if h1 * 60 + m1 <= mins <= h2 * 60 + m2:
            return True
    return False


# ============================= SIGNAL ENGINE =============================
def evaluate(candles5, candles15, candles30, candles60, now_utc, bypass_session=False):
    if len(candles5) < 60 or len(candles15) < 40 or len(candles30) < 30 or len(candles60) < 20:
        return {"state": "WARMING_UP"}

    price = candles5[-1]["c"]
    closes5 = [c["c"] for c in candles5]
    # MT5 chart is set to Method=Smoothed, Apply to=Close, Shift=0 - so use SMMA, not SMA.
    ma21 = smma(closes5, 21)
    ma50 = smma(closes5, 50)
    ma200 = smma(closes5, min(200, len(closes5)))
    rsi5 = rsi_series(closes5, RSI_PERIOD)
    rsi15 = rsi_series([c["c"] for c in candles15], RSI_PERIOD)
    rsi30 = rsi_series([c["c"] for c in candles30], RSI_PERIOD)
    atr5 = atr(candles5, ATR_PERIOD)

    # 5m ENTRY lines: full liveness filtering + bounce-weighted scoring.
    tl5up = detect_trendline(candles5, "resistance", TRENDLINE_RECENCY_CANDLES[5],
                             price=price, atr_val=atr5, apply_liveness=True)
    tl5dn = detect_trendline(candles5, "support", TRENDLINE_RECENCY_CANDLES[5],
                             price=price, atr_val=atr5, apply_liveness=True)
    # 15m also feeds the CORE trendline check, so it gets the same treatment.
    tl15up = detect_trendline(candles15, "resistance", TRENDLINE_RECENCY_CANDLES[15],
                              price=price, atr_val=atr5, apply_liveness=True)
    tl15dn = detect_trendline(candles15, "support", TRENDLINE_RECENCY_CANDLES[15],
                              price=price, atr_val=atr5, apply_liveness=True)
    # 30m is confluence-only: scored, but no liveness gating.
    tl30up = detect_trendline(candles30, "resistance", TRENDLINE_RECENCY_CANDLES[30], atr_val=atr5)
    tl30dn = detect_trendline(candles30, "support", TRENDLINE_RECENCY_CANDLES[30], atr_val=atr5)

    res_clusters, sup_clusters = detect_sr(candles60, SR_DEVIATION_USD)
    resistances = [c["avg"] for c in res_clusters]
    supports = [c["avg"] for c in sup_clusters]
    nearest_res = min((v for v in resistances if v > price), default=None)
    nearest_sup = max((v for v in supports if v < price), default=None)

    strike_type, strike_rev, strike_detail = detect_line_strike(candles5)
    div5 = detect_divergence(candles5, rsi5)
    div15 = detect_divergence(candles15, rsi15)
    div30 = detect_divergence(candles30, rsi30)

    real_in_session = in_session_window(now_utc)
    in_session = real_in_session or bypass_session
    ma_proximity = ma21 is not None and abs(price - ma21) <= FAST_MA_PROXIMITY_USD

    def eval_side(direction):
        conds = []  # CORE conditions - these drive the signal
        confluence = []  # CONFLUENCE conditions - bonus only, shown but never required

        pkt_now = toPKT_str(now_utc)
        if bypass_session and not real_in_session:
            session_label = f"Session · <code>{pkt_now} PKT</code> <i>(demo bypass)</i>"
        elif in_session:
            session_label = f"Session · <code>{pkt_now} PKT</code> <i>(7-16, 20-23)</i>"
        else:
            session_label = f"Session <i>— <code>{pkt_now} PKT</code> is outside 7-16 / 20-23</i>"
        conds.append(("session", session_label, in_session))
        ma_dist = abs(price - ma21) if ma21 is not None else None
        if ma21 is None:
            ma_label = "21MA <i>— not enough candles yet</i>"
        else:
            if ma_proximity:
                ma_label = (f"21MA <code>{ma21:.2f}</code> · <code>${ma_dist:.2f}</code> "
                            f"from price <i>(max ${FAST_MA_PROXIMITY_USD:.2f})</i>")
            else:
                ma_label = (f"21MA <code>{ma21:.2f}</code> · <code>${ma_dist:.2f}</code> away "
                            f"— <b>missed by ${ma_dist - FAST_MA_PROXIMITY_USD:.2f}</b>")
        conds.append(("ma_proximity", ma_label, ma_proximity))

        # --- CORE trendline: 5m and 15m only (30m demoted to confluence) ---
        # BUY = price breaks ABOVE resistance (line drawn from swing HIGHS), standard continuation.
        # SELL = price breaks BELOW support (line drawn from swing LOWS), standard continuation.
        core_tf = ([("5m", tl5up, candles5), ("15m", tl15up, candles15)] if direction == "buy" else
                   [("5m", tl5dn, candles5), ("15m", tl15dn, candles15)])
        tl_break_pass, tl_break_desc = False, "Trendline break <i>— no qualifying line broken</i>"
        tl_retest_pass, tl_retest_desc = False, "Trendline retest <i>— awaiting a break first</i>"
        # If nothing breaks, still report the closest line so the message shows how near it got.
        best_near = None
        for tf, line, cands in core_tf:
            if not line:
                continue
            last_len = len(cands)
            line_now = line_value_at(line, last_len - 1)
            if abs(line_now - price) > MAX_LINE_DISTANCE_USD:
                continue
            beyond = (price - line_now) if direction == "buy" else (line_now - price)
            if best_near is None or beyond > best_near[1]:
                best_near = (tf, beyond, line_now, line)
            if beyond < MIN_BREAK_USD:
                continue
            tl_break_pass = True
            tl_break_desc = (f"Broke {tf} line <code>{line_now:.2f}</code> by "
                             f"<code>${beyond:.2f}</code> <i>(min ${MIN_BREAK_USD:.2f})</i>")
            found, gap, lvl = retest_touch_found(
                cands, lambda i, ln=line: line_value_at(ln, i),
                TRENDLINE_RETEST_DEVIATION_USD, return_detail=True)
            if found:
                tl_retest_pass = True
                tl_retest_desc = (f"Retested {tf} line <code>{lvl:.2f}</code> "
                                  f"<i>(touched within ${TRENDLINE_RETEST_DEVIATION_USD:.2f})</i>")
            elif gap is not None:
                tl_retest_desc = (f"Retest · closest approach <code>${gap:.2f}</code> "
                                  f"— <b>missed by ${gap - TRENDLINE_RETEST_DEVIATION_USD:.2f}</b>")
            else:
                tl_retest_desc = f"Retest <i>— broke {tf}, price hasn't returned yet</i>"
            break
        if not tl_break_pass and best_near is not None:
            tf_n, beyond_n, lv_n, _ = best_near
            if beyond_n >= 0:
                tl_break_desc = (f"Break <code>{tf_n} {lv_n:.2f}</code> · only "
                                 f"<code>${beyond_n:.2f}</code> past — "
                                 f"<b>missed by ${MIN_BREAK_USD - beyond_n:.2f}</b>")
            else:
                tl_break_desc = (f"Break <code>{tf_n} {lv_n:.2f}</code> · price hasn't reached it "
                                 f"— <b>missed by ${abs(beyond_n) + MIN_BREAK_USD:.2f}</b>")
        conds.append(("trendline_break", tl_break_desc, tl_break_pass))
        conds.append(("trendline_retest", tl_retest_desc, tl_retest_pass))

        # --- CORE line-strike (5m) ---
        want_rev = "up" if direction == "buy" else "down"
        strike_match = strike_type is not None and strike_rev == want_rev
        if strike_match and strike_detail:
            d = strike_detail
            kind_txt = "3-line strike" if strike_type == "3-line" else "2-line strike <i>(partial)</i>"
            strike_desc = (f"{kind_txt} · closed <code>{d['close']:.2f}</code> past "
                           f"<code>{d['needed']:.2f}</code>, body <code>${d['body']:.2f}</code>")
        elif strike_match:
            strike_desc = "3-line strike" if strike_type == "3-line" else "2-line strike <i>(partial)</i>"
        elif (strike_detail and strike_detail.get("near_miss")
              and strike_rev == want_rev):
            strike_desc = (f"Strike · reversal candle closed <code>{strike_detail['close']:.2f}</code>, "
                           f"needed <code>{strike_detail['needed']:.2f}</code> — "
                           f"<b>missed by ${strike_detail['shortfall']:.2f}</b>")
        else:
            strike_desc = "Strike rejection <i>— no 3-of-a-kind reversal pattern</i>"
        conds.append(("strike", strike_desc, strike_match))

        # --- CORE divergence: 5m and 15m only (30m demoted to confluence) ---
        want_key = "bullish" if direction == "buy" else "bearish"
        core_hits = []
        for label, d in (("5m", div5), ("15m", div15)):
            if d[want_key]:
                det = d.get(f"{want_key}_detail")
                if det:
                    core_hits.append(
                        f"{label} {d[want_key]} · price <code>{det['p1']:.2f}→{det['p2']:.2f}</code> "
                        f"vs RSI <code>{det['r1']:.1f}→{det['r2']:.1f}</code>")
                else:
                    core_hits.append(f"{label} {d[want_key]}")
        if core_hits:
            div_desc = "RSI divergence · " + core_hits[0]
            for extra in core_hits[1:]:
                div_desc += f"\n      <i>+ {extra}</i>"
        else:
            div_desc = "RSI divergence <i>— none supporting this direction</i>"
        conds.append(("divergence", div_desc, len(core_hits) > 0))

        # ============= CONFLUENCE (30m trendline + 1H S&R) - bonus only =============
        # Messages spell out WHAT happened and WHY it backs this trade direction, so the
        # bonus section reads on its own without needing to remember the rules.
        dir_word = "BUY" if direction == "buy" else "SELL"
        tl30 = tl30up if direction == "buy" else tl30dn
        if tl30:
            line_now_30 = line_value_at(tl30, len(candles30) - 1)
            if abs(line_now_30 - price) <= MAX_LINE_DISTANCE_USD:
                broke_30 = (price > line_now_30 + MIN_BREAK_USD) if direction == "buy" else (price < line_now_30 - MIN_BREAK_USD)
                if broke_30:
                    tl30_side = "declining resistance" if direction == "buy" else "inclining support"
                    tl30_dir = "above" if direction == "buy" else "below"
                    confluence.append((
                        "confluence_tl30_break",
                        f"<b>30m</b> broke {tl30_dir} {tl30_side} @ <code>${line_now_30:.2f}</code>",
                        True))
                    if retest_touch_found(candles30, lambda i: line_value_at(tl30, i), TRENDLINE_RETEST_DEVIATION_USD):
                        confluence.append((
                            "confluence_tl30_retest",
                            "<b>30m</b> retested that line and held",
                            True))

        sr_break_pass, sr_break_desc = False, None
        sr_level = None
        if direction == "buy" and nearest_res is not None and price > nearest_res + MIN_BREAK_USD:
            sr_level, sr_break_pass = nearest_res, True
            sr_break_desc = f"<b>1H</b> broke above resistance @ <code>${nearest_res:.2f}</code>"
        if direction == "sell" and nearest_sup is not None and price < nearest_sup - MIN_BREAK_USD:
            sr_level, sr_break_pass = nearest_sup, True
            sr_break_desc = f"<b>1H</b> broke below support @ <code>${nearest_sup:.2f}</code>"
        if sr_break_pass:
            confluence.append(("confluence_sr_break", sr_break_desc, True))
            if retest_touch_found(candles60, lambda i: sr_level, SR_DEVIATION_USD):
                confluence.append((
                    "confluence_sr_retest",
                    f"<b>1H</b> retested <code>${sr_level:.2f}</code> and held",
                    True))

        if div30[want_key]:
            div_kind = div30[want_key]
            div_meaning = ("momentum weakening against the old move" if div_kind == "regular"
                           else "momentum still favouring continuation")
            confluence.append((
                "confluence_div30",
                f"<b>30m</b> {div_kind} RSI divergence <i>({div_meaning})</i>",
                True))

        core_passed = sum(1 for _, _, p in conds if p)
        core_total = len(conds)
        confluence_count = len(confluence)  # confluence list only ever contains passed items; shown as bonus info only

        # Full signal requires ALL 6 core conditions - confluence never gates or delays this,
        # since 30m/1H structure breaks far less often than 5m/15m and shouldn't hold up real trades.
        full_signal = core_passed >= FULL_SIGNAL_MIN_CONDITIONS
        pre_signal = core_passed >= PRE_SIGNAL_MIN_CONDITIONS
        strong_pre_signal = core_passed >= FULL_SIGNAL_MIN_CONDITIONS - 1  # one condition away - worth flagging clearly

        return {
            "dir": direction, "conds": conds, "confluence": confluence,
            "passed": core_passed, "total": core_total, "confluence_count": confluence_count,
            "all_pass": full_signal, "nearly_all": pre_signal, "strong_pre": strong_pre_signal,
        }

    buy_side, sell_side = eval_side("buy"), eval_side("sell")
    chosen = buy_side if buy_side["passed"] >= sell_side["passed"] else sell_side

    state = "NO_TRADE"
    if in_session:
        if chosen["all_pass"]:
            state = "BUY" if chosen["dir"] == "buy" else "SELL"
        elif chosen["nearly_all"]:
            state = "PRE_BUY" if chosen["dir"] == "buy" else "PRE_SELL"

    entry = sl = tp = None
    obstacles = []
    if atr5 and state != "NO_TRADE" and state != "WARMING_UP":
        dist = atr5 * 2
        entry = price
        if "BUY" in state:
            sl, tp = price - dist, price + dist * RISK_REWARD
        else:
            sl, tp = price + dist, price - dist * RISK_REWARD

        # Room-to-target check: find the NEAREST qualifying trendline on 15m/30m/1H sitting
        # between entry and target. No liveness filter here - a dormant wall still blocks price.
        direction_is_buy = "BUY" in state
        obs_side = "resistance" if direction_is_buy else "support"
        obstacles = []
        for label, cands, win in (("15m", candles15, TRENDLINE_RECENCY_CANDLES[15]),
                                   ("30m", candles30, TRENDLINE_RECENCY_CANDLES[30]),
                                   ("1H", candles60, TRENDLINE_RECENCY_CANDLES[60])):
            ob = detect_nearest_obstacle_line(cands, obs_side, win, entry, tp)
            if ob:
                obstacles.append(
                    f"<b>{label}</b> {obs_side} <code>{ob['line_now']:.2f}</code> · "
                    f"<code>${ob['dist_from_entry']:.2f}</code> from entry, "
                    f"<code>${abs(tp - ob['line_now']):.2f}</code> short of target")

    is_risky = len(obstacles) > 0 and state in ("BUY", "SELL")

    return {
        "state": state, "price": price, "entry": entry, "sl": sl, "tp": tp,
        "conds": chosen["conds"], "passed": chosen["passed"], "total": chosen["total"],
        "confluence": chosen["confluence"], "confluence_count": chosen["confluence_count"],
        "strong_pre": chosen["strong_pre"],
        "obstacles": obstacles, "is_risky": is_risky,
        "indicators": {"ma21": ma21, "ma50": ma50, "ma200": ma200,
                       "rsi": rsi5[-1] if rsi5 else None, "atr": atr5, "in_session": in_session},
    }


# ============================= TELEGRAM =============================
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping send. Message would have been:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
        if not r.ok:
            print("Telegram send failed:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send error:", e)



RULE = "━━━━━━━━━━━━━━━━━━━━"


def _meter(passed, total):
    """Compact filled/empty bar for the setup strength."""
    return "█" * passed + "░" * max(0, total - passed)


def _levels_block(res):
    """Monospaced entry/stop/target block so the numbers line up on a phone."""
    if res.get("entry") is None:
        return []
    entry, sl, tp = res["entry"], res["sl"], res["tp"]
    risk, reward = abs(entry - sl), abs(tp - entry)
    return [
        f"<code>ENTRY   {entry:>9.2f}</code>",
        f"<code>STOP    {sl:>9.2f}   -{risk:.2f}</code>",
        f"<code>TARGET  {tp:>9.2f}   +{reward:.2f}</code>",
    ]


def _record_line(tp_hits, sl_hits, label="RECORD"):
    total = tp_hits + sl_hits
    if total == 0:
        return f"<code>{label}</code>  <i>no closed trades yet</i>"
    rate = (tp_hits / total) * 100
    return f"<code>{label}</code>  {tp_hits}W · {sl_hits}L · <b>{rate:.0f}%</b>"


def _conditions_block(res):
    passed, total = res.get("passed", 0), res.get("total", 0)
    if not total:
        return ["<i>Still building candle history — no evaluation yet.</i>"]
    out = [f"<b>SETUP</b>  {passed}/{total}  <code>{_meter(passed, total)}</code>", ""]
    for _, label, ok in res.get("conds", []):
        out.append(f"{'✅' if ok else '▫️'} {label}")
    return out


def _confluence_block(res):
    if not res.get("confluence"):
        return []
    out = ["", "<b>✨ BIGGER PICTURE</b> <i>(bonus only)</i>"]
    for _, label, _ in res["confluence"]:
        out.append(f"   {label}")
    return out


def _data_warning_block(res):
    """Data cautions shown ON the signal - never a reason to suppress it."""
    if not res.get("data_warnings"):
        return []
    out = ["", "<b>⚠️ DATA CHECK</b>"]
    for w in res["data_warnings"]:
        out.append(f"   {w}")
    return out


def build_message(res):
    """Pre-signal / demo status message."""
    state = res["state"]
    if state == "BUY":
        head = "🟢 <b>BUY SIGNAL</b>"
    elif state == "SELL":
        head = "🔴 <b>SELL SIGNAL</b>"
    elif state == "PRE_BUY":
        head = ("🟠 <b>STRONG PRE-SIGNAL</b> · BUY" if res.get("strong_pre")
                else "🟡 <b>PRE-SIGNAL</b> · BUY")
    elif state == "PRE_SELL":
        head = ("🟠 <b>STRONG PRE-SIGNAL</b> · SELL" if res.get("strong_pre")
                else "🟡 <b>PRE-SIGNAL</b> · SELL")
    elif state == "WARMING_UP":
        head = "⏳ <b>WARMING UP</b>"
    else:
        head = "⚪️ <b>NO TRADE</b>"

    price = res.get("price")
    price_txt = f"<code>${price:.2f}</code>" if price is not None else "<i>n/a</i>"
    lines = [head, RULE, f"<b>XAU/USD</b>   {price_txt}"]

    levels = _levels_block(res)
    if levels:
        lines += [RULE] + levels

    lines += _data_warning_block(res)
    lines += [RULE, ""] + _conditions_block(res) + _confluence_block(res)
    return "\n".join(lines)


def build_confirmed_signal_message(res, signal_number, tp_hits, sl_hits):
    """Confirmed BUY/SELL - carries its signal number and running record."""
    if res.get("is_risky"):
        head = (f"🟢 <b>BUY</b> <i>(Risky)</i> · <b>#{signal_number}</b>" if res["state"] == "BUY"
                else f"🔴 <b>SELL</b> <i>(Risky)</i> · <b>#{signal_number}</b>")
    else:
        head = (f"🟢 <b>BUY SIGNAL</b> · <b>#{signal_number}</b>" if res["state"] == "BUY"
                else f"🔴 <b>SELL SIGNAL</b> · <b>#{signal_number}</b>")

    lines = [head, RULE, f"<b>XAU/USD</b>   <code>${res['price']:.2f}</code>"]

    levels = _levels_block(res)
    if levels:
        lines += [RULE] + levels

    if res.get("obstacles"):
        lines += [RULE, "<b>⚠️ RISK</b>"]
        for o in res["obstacles"]:
            lines.append(f"   {o}")

    lines += _data_warning_block(res)
    lines += [RULE, ""] + _conditions_block(res) + _confluence_block(res)
    label = "RISKY" if res.get("is_risky") else "RECORD"
    lines += ["", RULE, _record_line(tp_hits, sl_hits, label)]
    return "\n".join(lines)


def build_outcome_message(trade, outcome, tp_hits, sl_hits):
    """Follow-up sent once a tracked signal resolves."""
    won = outcome == "TP"
    kind = trade.get("kind", "confirmed")
    kind_label = {"confirmed": "Signal", "risky": "Risky signal", "pre": "Pre-signal"}.get(kind, "Signal")
    head = (f"✅ <b>TARGET HIT</b> · {kind_label} #{trade['signal_number']}" if won
            else f"❌ <b>STOPPED OUT</b> · {kind_label} #{trade['signal_number']}")
    moved = abs(trade["tp"] - trade["entry"]) if won else -abs(trade["entry"] - trade["sl"])
    record_label = {"confirmed": "RECORD", "risky": "RISKY", "pre": "PRE"}.get(kind, "RECORD")
    return "\n".join([
        head,
        RULE,
        f"<code>{trade['dir']:<6} {trade['entry']:>9.2f}</code>",
        f"<code>RESULT {moved:>+9.2f}</code>",
        RULE,
        _record_line(tp_hits, sl_hits, record_label),
    ])




# ============================= TRADE OUTCOME TRACKING =============================
def resolve_trade(trade, candles5):
    """Walk candles chronologically after the trade opened to see whether TP or SL was
    hit first. If a single candle's range covers both levels (ambiguous intrabar path),
    conservatively counts it as SL hit rather than assuming the best case."""
    opened_ms = trade["opened_at_ms"]
    direction = trade["dir"]
    tp, sl = trade["tp"], trade["sl"]
    for c in candles5:
        if c["t"] <= opened_ms:
            continue
        if direction == "BUY":
            hit_tp, hit_sl = c["h"] >= tp, c["l"] <= sl
        else:
            hit_tp, hit_sl = c["l"] <= tp, c["h"] >= sl
        if hit_tp and hit_sl:
            return "SL"
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return None  # not resolved yet


# A trade older than this can never resolve - the data feed only retains ~48h, so once
# it scrolls out of the window we'd re-scan it forever and state.json would grow unbounded.
TRADE_EXPIRY_HOURS = 36


def check_data_sanity(points, spot, candles5, state_store, now_utc):
    """Inspect the incoming data before trusting it.

    Returns (fatal_reason, warnings).
      fatal_reason -> a string only when the data is genuinely UNUSABLE (nothing to
                      evaluate anyway, so no trade can be lost by stopping).
      warnings     -> list of strings attached to the signal as a caution. These NEVER
                      block a signal: in a break-and-retest strategy a fast move is the
                      setup, not an error, so refusing to trade on it would be backwards.
    """
    warnings = []

    # ---- FATAL: nothing usable to evaluate ----
    if not points:
        return "price feed returned no data", warnings
    if not candles5:
        return "could not build any candles from the feed", warnings
    latest = candles5[-1]["c"]
    if latest is None or latest <= 0:
        return f"latest price is invalid ({latest})", warnings

    # ---- WARN: stale feed ----
    newest_ms = candles5[-1]["t"]
    age_min = (now_utc.timestamp() * 1000 - newest_ms) / 60000.0
    if age_min > DATA_STALE_WARN_MINUTES:
        warnings.append(f"feed looks stale — newest candle is {age_min:.0f} min old")

    # ---- WARN: spot vs candle disagreement ----
    spot_price = (spot or {}).get("spot_usd_oz")
    if spot_price:
        diff = abs(spot_price - latest)
        if diff > DATA_SPOT_MISMATCH_WARN:
            warnings.append(
                f"feed disagrees with itself — spot ${spot_price:.2f} vs candle ${latest:.2f} "
                f"(${diff:.2f} apart)")

    # ---- WARN: large jump since the previous scan ----
    prev = state_store.get("last_seen_price")
    if prev:
        pct = abs(latest - prev) / prev * 100.0
        if pct > DATA_JUMP_WARN_PCT:
            warnings.append(
                f"price moved {pct:.1f}% since last scan (${prev:.2f} → ${latest:.2f}) — "
                f"could be a real move or bad data, verify on MT5")

    # ---- WARN: thin history ----
    if len(candles5) < 100:
        warnings.append(f"only {len(candles5)} 5m candles available — indicators may be unreliable")

    return None, warnings


def should_send_warning(state_store, now_utc):
    """Rate-limit warning alerts so a flaky feed can't spam you every 5 minutes."""
    last = state_store.get("last_warn_ms")
    now_ms = now_utc.timestamp() * 1000
    if last and (now_ms - last) < DATA_WARN_COOLDOWN_MINUTES * 60000:
        return False
    state_store["last_warn_ms"] = now_ms
    return True


def build_heartbeat_message(res, state_store, now_utc):
    """Daily 'I'm alive' summary so silence is never ambiguous."""
    price = res.get("price")
    price_txt = f"<code>${price:.2f}</code>" if price else "<i>n/a</i>"
    lines = [
        "💚 <b>DAILY CHECK-IN</b>",
        RULE,
        f"<b>XAU/USD</b>   {price_txt}",
        f"<code>{toPKT_str(now_utc)} PKT</code>",
        RULE,
        _record_line(state_store.get("tp_hits", 0), state_store.get("sl_hits", 0), "SIGNALS"),
        _record_line(state_store.get("risky_tp_hits", 0), state_store.get("risky_sl_hits", 0), "RISKY  "),
        _record_line(state_store.get("pre_tp_hits", 0), state_store.get("pre_sl_hits", 0), "PRE    "),
        RULE,
        f"<code>FIRED  </code>  {state_store.get('signal_counter', 0)} signals · "
        f"{state_store.get('risky_counter', 0)} risky · {state_store.get('pre_counter', 0)} pre",
        f"<code>OPEN   </code>  {len(state_store.get('pending_trades', []))} trades awaiting TP/SL",
    ]
    return "\n".join(lines)


def maybe_send_heartbeat(res, state_store, now_utc):
    """Fire once per calendar day, regardless of how many loop iterations run."""
    pkt = now_utc + timedelta(hours=5)
    today = pkt.strftime("%Y-%m-%d")
    if state_store.get("last_heartbeat_date") == today:
        return
    if pkt.hour < HEARTBEAT_HOUR_PKT:
        return
    send_telegram(build_heartbeat_message(res, state_store, now_utc))
    state_store["last_heartbeat_date"] = today


def process_pending_trades(candles5, state_store, now_utc=None):
    pending = state_store.get("pending_trades", [])
    still_pending = []
    now_ms = int((now_utc or datetime.now(timezone.utc)).timestamp() * 1000)
    expiry_ms = TRADE_EXPIRY_HOURS * 3600 * 1000
    for trade in pending:
        outcome = resolve_trade(trade, candles5)
        if outcome is None:
            age = now_ms - trade.get("opened_at_ms", now_ms)
            if age > expiry_ms:
                # Neither TP nor SL was reached inside the data window - drop it rather
                # than tracking it forever. Not counted as a win or a loss.
                print(f"Expiring unresolved trade #{trade.get('signal_number')} "
                      f"({trade.get('kind')}) after {age/3600000:.0f}h")
                continue
            still_pending.append(trade)
            continue
        kind = trade.get("kind", "confirmed")
        if kind == "risky":
            tp_key, sl_key = "risky_tp_hits", "risky_sl_hits"
        elif kind == "pre":
            tp_key, sl_key = "pre_tp_hits", "pre_sl_hits"
        else:
            tp_key, sl_key = "tp_hits", "sl_hits"

        if outcome == "TP":
            state_store[tp_key] = state_store.get(tp_key, 0) + 1
        else:
            state_store[sl_key] = state_store.get(sl_key, 0) + 1
        send_telegram(build_outcome_message(trade, outcome,
                                            state_store.get(tp_key, 0), state_store.get(sl_key, 0)))
    state_store["pending_trades"] = still_pending


# ============================= STATE PERSISTENCE =============================
def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.setdefault("last_state", "NONE")
    # Three independent trackers: confirmed signals, risky signals, and pre-signals
    data.setdefault("signal_counter", 0)
    data.setdefault("tp_hits", 0)
    data.setdefault("sl_hits", 0)
    data.setdefault("risky_counter", 0)
    data.setdefault("risky_tp_hits", 0)
    data.setdefault("risky_sl_hits", 0)
    data.setdefault("pre_counter", 0)
    data.setdefault("pre_tp_hits", 0)
    data.setdefault("pre_sl_hits", 0)
    data.setdefault("pending_trades", [])
    return data


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================= MAIN =============================
def main():
    now_utc = datetime.now(timezone.utc)
    # A manual run should send one confirmation message, not one per loop iteration.
    # The workflow sets SCAN_ITERATION; only the first pass counts as the demo run.
    scan_iteration = os.environ.get("SCAN_ITERATION", "1")
    is_manual_run = (os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
                     and scan_iteration == "1")

    in_session = in_session_window(now_utc)
    state_store = load_state()

    # The heartbeat should fire even outside session hours, so fetch first and gate later.
    try:
        points = fetch_intraday()
        spot = fetch_spot()
    except Exception as e:
        print("Data fetch failed:", e)
        # A persistent outage is worth knowing about; the cooldown stops it spamming.
        if should_send_warning(state_store, now_utc):
            send_telegram(f"⚠️ <b>Data feed unreachable</b>\n{RULE}\n<i>{e}</i>\n"
                          f"Scanning will resume automatically when the feed returns.")
            save_state(state_store)
        sys.exit(0)  # don't fail the workflow on a transient API hiccup

    candles5 = build_candles(points, 5)
    candles15 = build_candles(points, 15)
    candles30 = build_candles(points, 30)
    candles60 = build_candles(points, 60)

    # --- Data sanity: fatal only when there is genuinely nothing to evaluate ---
    fatal, data_warnings = check_data_sanity(points, spot, candles5, state_store, now_utc)
    if fatal:
        print("Data unusable:", fatal)
        if should_send_warning(state_store, now_utc):
            send_telegram(f"⚠️ <b>Data problem — scanning paused</b>\n{RULE}\n<i>{fatal}</i>\n"
                          f"No signals will fire until this clears.")
        save_state(state_store)
        return

    if candles5:
        state_store["last_seen_price"] = candles5[-1]["c"]

    res = evaluate(candles5, candles15, candles30, candles60, now_utc, bypass_session=is_manual_run)
    res["data_warnings"] = data_warnings

    # Daily check-in runs regardless of session, so silence is never ambiguous.
    maybe_send_heartbeat(res, state_store, now_utc)

    if not in_session and not is_manual_run:
        print("Outside trading session window (PKT). Skipping evaluation.")
        save_state(state_store)
        return

    if data_warnings:
        print("Data warnings:", "; ".join(data_warnings))
    print("Evaluated state:", res.get("state"), "| spot:", spot.get("spot_usd_oz"))

    if res.get("state") == "WARMING_UP":
        if is_manual_run:
            send_telegram("⏳ Demo run: still building candle history from live ticks — "
                           "try again in a few minutes once more data has accumulated.")
        print("Still warming up, not enough candle history yet.")
        return

    # Check any open confirmed signals for TP/SL resolution first, regardless of current state.
    if not is_manual_run:
        process_pending_trades(candles5, state_store, now_utc)

    last_state = state_store.get("last_state", "NONE")
    current_state = res.get("state")

    if is_manual_run:
        # Manual/demo run: always send a status message so you can confirm Telegram delivery,
        # regardless of whether the state actually changed. Demo runs do NOT register signals
        # or affect the TP/SL tally, so testing never pollutes your real track record.
        prefix = "🧪 <b>Demo run</b>\n\n" if current_state == "NO_TRADE" else ""
        send_telegram(prefix + build_message(res))
        print("Manual run: notification sent regardless of state change.")
        return

    if current_state in ("BUY", "SELL") and current_state != last_state:
        is_risky = res.get("is_risky", False)
        if is_risky:
            counter_key, tp_key, sl_key, kind = "risky_counter", "risky_tp_hits", "risky_sl_hits", "risky"
        else:
            counter_key, tp_key, sl_key, kind = "signal_counter", "tp_hits", "sl_hits", "confirmed"

        tp_hits, sl_hits = state_store.get(tp_key, 0), state_store.get(sl_key, 0)
        signal_number = state_store.get(counter_key, 0) + 1
        state_store[counter_key] = signal_number

        send_telegram(build_confirmed_signal_message(res, signal_number, tp_hits, sl_hits))

        state_store.setdefault("pending_trades", []).append({
            "signal_number": signal_number, "dir": current_state, "kind": kind,
            "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
            "opened_at_ms": int(now_utc.timestamp() * 1000),
        })
        state_store["last_state"] = current_state
        state_store["last_signal_time"] = now_utc.isoformat()
        save_state(state_store)

    elif current_state in ("PRE_BUY", "PRE_SELL") and current_state != last_state:
        # Pre-signals get their own independent counter and TP/SL tracker.
        pre_number = state_store.get("pre_counter", 0) + 1
        state_store["pre_counter"] = pre_number
        pre_tp, pre_sl = state_store.get("pre_tp_hits", 0), state_store.get("pre_sl_hits", 0)

        msg = build_message(res)
        msg += "\n\n" + RULE + "\n" + f"<code>PRE #{pre_number}</code>  " + \
               _record_line(pre_tp, pre_sl, "PRE").split("</code>", 1)[-1].strip()
        send_telegram(msg)

        if res.get("entry") is not None:
            state_store.setdefault("pending_trades", []).append({
                "signal_number": pre_number, "dir": "BUY" if current_state == "PRE_BUY" else "SELL",
                "kind": "pre",
                "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
                "opened_at_ms": int(now_utc.timestamp() * 1000),
            })
        state_store["last_state"] = current_state
        save_state(state_store)

    elif current_state == "NO_TRADE" and last_state != "NONE":
        state_store["last_state"] = "NONE"
        save_state(state_store)

    else:
        print("No state change, no notification sent.")
        save_state(state_store)  # still persist any pending-trade resolutions from above


if __name__ == "__main__":
    main()
