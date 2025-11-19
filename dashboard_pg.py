#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:40:04 2025

@author: tserennyamsanjsuren
"""



from datetime import date as _date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from sqlalchemy import create_engine, text, event
from pandas.api.types import CategoricalDtype

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Corridor Flight Dashboard (ET, PostgreSQL)",
    layout="wide",
)

# Make the main content wider & improve tab label visibility
st.markdown(
    """
    <style>
      .block-container {
        max-width: 2000px;
        padding-top: 1rem;
        padding-bottom: 1rem;
      }
      .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 700;
        padding: 10px 16px;
      }
      .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #ff5252 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CONSTANTS / CONNECTION
# =========================
DATABASE_URL = "postgresql+psycopg2://airdash:KW-triO123@127.0.0.1:65432/airdash_db"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    pool_recycle=1800,
)

@event.listens_for(engine, "connect")
def _set_stmt_timeout(dbapi_conn, conn_record):
    cur = dbapi_conn.cursor()
    cur.execute("SET statement_timeout = '120s';")
    cur.execute("SET work_mem = '64MB';")
    cur.close()

# Corridor: Baltimore → NYC, FL200–FL430
LAT_MIN, LAT_MAX = 38.9, 41.2
LON_MIN, LON_MAX = -77.6, -72.0
ALT_MIN_FT, ALT_MAX_FT = 20000, 43000

# Heavy families
HEAVY_FAMILIES = ('B777', 'B787', 'A330', 'A330neo', 'A350', 'B75/76')

# =========================
# HELPERS
# =========================
def table_exists(schema: str, name: str) -> bool:
    q = text(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = :s AND table_name = :n
        ) AS ok
        """
    )
    with engine.connect() as cn:
        return bool(pd.read_sql(q, cn, params={"s": schema, "n": name}).iloc[0]["ok"])


def columns_exist(schema: str, table: str, needed: list[str]) -> bool:
    q = text("""
        SELECT LOWER(column_name) AS col
        FROM information_schema.columns
        WHERE table_schema = :s AND table_name = :t
    """)
    with engine.connect() as cn:
        cols = set(pd.read_sql(q, cn, params={"s": schema, "t": table})["col"])
    return all(c.lower() in cols for c in needed)


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Great-circle initial bearing (deg)."""
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)
    θ = np.degrees(np.arctan2(y, x))
    return (θ + 360.0) % 360.0


def _unwrap_deg(deg_series: pd.Series) -> np.ndarray:
    """Unwrap heading to avoid 360/0 jumps (returns radians unwrapped -> deg)."""
    return np.degrees(np.unwrap(np.radians(deg_series.to_numpy())))
@st.cache_data(ttl=3600, show_spinner=False)
def get_wind_hourly(start_date_et):
    """
    Hourly wind effect from core.corridor_wind_hourly.

    Uses:
      - approx_headwind_ms  (positive = headwind, negative = tailwind)
      - approx_crosswind_ms
      - avg_wind_dir_deg    (wind direction in degrees)

    Returns all rows from the chosen start_date_et onward.
    """
    sql = text("""
        SELECT
            hour_et,
            approx_headwind_ms,
            approx_crosswind_ms,
            avg_wind_speed_ms,
            avg_gs_kt,
            avg_wind_dir_deg
        FROM core.corridor_wind_hourly
        WHERE hour_et >= :start
        ORDER BY hour_et;
    """)

    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"start": start_date_et})

    if df.empty:
        return df

    # Convert to knots
    factor = 1.94384
    df["headwind_kt"]   = df["approx_headwind_ms"]  * factor
    df["crosswind_kt"]  = df["approx_crosswind_ms"] * factor
    df["wind_speed_kt"] = df["avg_wind_speed_ms"]   * factor

    # For convenience
    df["hour_et"] = pd.to_datetime(df["hour_et"])
    df["day_et"] = df["hour_et"].dt.date

    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_top_aircraft_types(start_day, end_day, limit: int = 10) -> pd.DataFrame:
    """
    Top aircraft types in the corridor between start_day and end_day (inclusive),
    based on distinct callsigns in core.corridor_rows.

    Returns columns: typecode, flights, share_pct
    """
    # Require corridor_rows; otherwise, return empty
    if not table_exists("core", "corridor_rows"):
        return pd.DataFrame()

    sql = text("""
        WITH base AS (
          SELECT DISTINCT day_et, callsign, icao24
          FROM core.corridor_rows
          WHERE day_et BETWEEN :s AND :e
        ),
        typed AS (
          SELECT
            b.day_et,
            b.callsign,
            COALESCE(NULLIF(ot.typecode,''), NULLIF(im.typecode,'')) AS typecode
          FROM base b
          LEFT JOIN core.os_aircraft   ot ON ot.icao24 = b.icao24
          LEFT JOIN core.icao_type_map im ON im.icao24 = b.icao24
        )
        SELECT
          typecode,
          COUNT(DISTINCT callsign) AS flights
        FROM typed
        WHERE typecode IS NOT NULL AND typecode <> ''
        GROUP BY typecode
        ORDER BY flights DESC
        LIMIT :lim;
    """)
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"s": start_day, "e": end_day, "lim": int(limit)})

    if df.empty:
        return df

    total = df["flights"].sum()
    df["share_pct"] = (df["flights"] / total * 100.0).round(1)
    return df

# =========================
# FUEL HELPERS
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fuel_last_n_days(n: int = 30) -> pd.DataFrame:
    """
    Fallback: pull the last N days from core.mv_daily_fuel.
    Returns ascending by day (oldest → newest).
    """
    sql = text("""
        SELECT day_et, fuel_kg, co2_kg
        FROM core.mv_daily_fuel
        ORDER BY day_et DESC
        LIMIT :n
    """)
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"n": int(n)})
    return df.sort_values("day_et").reset_index(drop=True)


def refresh_mv_fuel() -> None:
    """
    Try to refresh the materialized view. Uses CONCURRENTLY when possible,
    and falls back to a normal refresh if that fails.
    """
    try:
        with engine.begin() as cn:
            cn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY core.mv_daily_fuel;"))
    except Exception:
        with engine.begin() as cn:
            cn.execute(text("REFRESH MATERIALIZED VIEW core.mv_daily_fuel;"))

# =========================
# CACHED QUERIES
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def get_kpi(start_date_et: _date) -> pd.DataFrame:
    """Reads from core.mv_daily_kpis(day_et date, flights_unique int)."""
    sql = text(
        """
        SELECT day_et, flights_unique
        FROM core.mv_daily_kpis
        WHERE day_et >= :start
        ORDER BY day_et
        """
    )
    with engine.connect() as cn:
        return pd.read_sql(sql, cn, params={"start": start_date_et})


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_flights(limit: int = 20) -> pd.DataFrame:
    candidates = [
        "core.mv_corridor_recent_24h",
        "core.v_corridor_recent_24h",
        "core.v_recent_24h",
    ]
    with engine.connect() as cn:
        for v in candidates:
            try:
                sql = text(f"SELECT * FROM {v} ORDER BY ts_et DESC LIMIT :lim")
                return pd.read_sql(sql, cn, params={"lim": int(limit)})
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_flights_for_day(day_et: str, limit: int = 20) -> pd.DataFrame:
    """
    Return the most recent N rows for a specific ET calendar day.
    Prefers core.corridor_rows and falls back to core.flights with corridor bounds.
    """
    if table_exists("core", "corridor_rows"):
        sql = text("""
            SELECT
              (ts_utc AT TIME ZONE 'America/New_York') AS ts_et,
              icao24, callsign, lat, lon, alt_ft, groundspeed_kt, track_deg
            FROM core.corridor_rows
            WHERE day_et = :d
            ORDER BY ts_utc DESC
            LIMIT :lim
        """)
        params = {"d": day_et, "lim": int(limit)}
    else:
        sql = text(f"""
            SELECT
              (ts_utc AT TIME ZONE 'America/New_York') AS ts_et,
              icao24, callsign, lat, lon, alt_ft, groundspeed_kt, track_deg
            FROM core.flights
            WHERE (ts_utc AT TIME ZONE 'America/New_York')::date = :d
              AND lat BETWEEN {LAT_MIN} AND {LAT_MAX}
              AND lon BETWEEN {LON_MIN} AND {LON_MAX}
              AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}
            ORDER BY ts_utc DESC
            LIMIT :lim
        """)
        params = {"d": day_et, "lim": int(limit)}

    with engine.connect() as cn:
        return pd.read_sql(sql, cn, params=params)


@st.cache_data(ttl=3600, show_spinner=False)
def list_days_et() -> list[str]:
    if table_exists("core", "corridor_rows"):
        has_day = columns_exist("core", "corridor_rows", ["day_et"])
        expr = "day_et" if has_day else "(ts_utc AT TIME ZONE 'America/New_York')::date"
        sql = text(f"SELECT DISTINCT {expr} AS d FROM core.corridor_rows ORDER BY d")
    else:
        sql = text(f"""
            SELECT DISTINCT ((ts_utc AT TIME ZONE 'America/New_York')::date) AS d
            FROM core.flights
            WHERE lat BETWEEN {LAT_MIN} AND {LAT_MAX}
              AND lon BETWEEN {LON_MIN} AND {LON_MAX}
              AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}
            ORDER BY d
        """)
    with engine.connect() as cn:
        return pd.read_sql(sql, cn)["d"].astype(str).tolist()


@st.cache_data(ttl=900, show_spinner=False)
def get_top_airlines(day_et: str, limit: int = 20) -> pd.DataFrame:
    """Top N airlines by distinct callsigns for the ET day."""
    base = "core.corridor_rows" if table_exists("core", "corridor_rows") else "core.flights"
    day_filter = "day_et = :d" if base.endswith("corridor_rows") else (
        f"(ts_utc AT TIME ZONE 'America/New_York')::date = :d "
        f"AND lat BETWEEN {LAT_MIN} AND {LAT_MAX} "
        f"AND lon BETWEEN {LON_MIN} AND {LON_MAX} "
        f"AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}"
    )
    sql = text(
        f"""
        SELECT COALESCE(NULLIF(SUBSTRING(UPPER(callsign) FROM '^[A-Z]{{3}}'), ''),'PRIV') AS airline,
               COUNT(DISTINCT callsign) AS flights
        FROM {base}
        WHERE {day_filter}
        GROUP BY 1
        ORDER BY flights DESC
        LIMIT :lim
        """
    )
    with engine.connect() as cn:
        return pd.read_sql(sql, cn, params={"d": day_et, "lim": int(limit)})


@st.cache_data(ttl=900, show_spinner=False)
def get_hourly_counts(day_et: str) -> pd.DataFrame:
    base = "core.corridor_rows" if table_exists("core", "corridor_rows") else "core.flights"
    use_day_col = base.endswith("corridor_rows") and columns_exist("core", "corridor_rows", ["day_et"])
    day_filter = "day_et = :d" if use_day_col else "((ts_utc AT TIME ZONE 'America/New_York')::date = :d)"
    geo = "" if base.endswith("corridor_rows") else (
        f" AND lat BETWEEN {LAT_MIN} AND {LAT_MAX}"
        f" AND lon BETWEEN {LON_MIN} AND {LON_MAX}"
        f" AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}"
    )
    sql = text(f"""
        SELECT date_trunc('hour', ts_utc) AS hour_utc,
               COUNT(DISTINCT callsign) AS flights
        FROM {base}
        WHERE {day_filter}{geo}
        GROUP BY 1
        ORDER BY 1
    """)
    with engine.connect() as cn:
        return pd.read_sql(sql, cn, params={"d": day_et})


@st.cache_data(ttl=1800, show_spinner=False)
def get_map_sample(day_et: str, target_points: int = 100_000) -> pd.DataFrame:
    """Sampled lat/lon/alt for map rendering."""
    base = "core.corridor_rows" if table_exists("core", "corridor_rows") else "core.flights"
    day_filter = "day_et = :d" if base.endswith("corridor_rows") else (
        f"(ts_utc AT TIME ZONE 'America/New_York')::date = :d "
        f"AND lat BETWEEN {LAT_MIN} AND {LAT_MAX} "
        f"AND lon BETWEEN {LON_MIN} AND {LON_MAX} "
        f"AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}"
    )
    sql = text(
        f"""
        WITH day AS (
          SELECT ts_utc, lat::real AS lat, lon::real AS lon, callsign, alt_ft
          FROM {base}
          WHERE {day_filter}
        ),
        cnt AS (SELECT COUNT(*) AS n FROM day),
        param AS (
          SELECT GREATEST(1, CEIL((SELECT n FROM cnt)::numeric / :target)::int) AS modv
        )
        SELECT d.ts_utc, d.lat, d.lon, d.callsign, d.alt_ft
        FROM day d, param p
        WHERE MOD(ABS(hashtext(d.callsign || to_char(date_trunc('minute', d.ts_utc), 'YYYYMMDDHH24MI'))), p.modv) = 0
        ORDER BY d.ts_utc
        LIMIT :lim
        """
    )
    with engine.connect() as cn:
        return pd.read_sql(sql, cn, params={"d": day_et, "target": int(target_points), "lim": int(target_points)})


@st.cache_data(ttl=1800, show_spinner=False)
def get_rows_for_maneuvers(day_et: str, max_rows: int = 400_000) -> pd.DataFrame:
    """
    Unsampled rows for maneuver detection (ordered by callsign,time).
    Needs: core.corridor_rows(day_et, ts_utc, lat, lon, alt_ft, callsign [, track_deg])
    """
    base = "core.corridor_rows" if table_exists("core", "corridor_rows") else "core.flights"
    day_filter = "day_et = :d" if base.endswith("corridor_rows") else (
        f"(ts_utc AT TIME ZONE 'America/New_York')::date = :d "
        f"AND lat BETWEEN {LAT_MIN} AND {LAT_MAX} "
        f"AND lon BETWEEN {LON_MIN} AND {LON_MAX} "
        f"AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}"
    )
    sql = text(
        f"""
        SELECT ts_utc, lat::real AS lat, lon::real AS lon, alt_ft::real AS alt_ft,
               callsign
               {", track_deg::real AS track_deg" if table_exists("core","corridor_rows") else ""}
        FROM {base}
        WHERE {day_filter}
        ORDER BY callsign, ts_utc
        LIMIT :lim
        """
    )
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"d": day_et, "lim": int(max_rows)})
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def get_alt_time_sample(day_et: str, target_points: int = 150_000) -> pd.DataFrame:
    """
    Sample (ts_utc, alt_ft) points for the Altitude vs Time scatter.
    """
    base = "core.corridor_rows" if table_exists("core", "corridor_rows") else "core.flights"
    day_filter = "day_et = :d" if base.endswith("corridor_rows") else (
        f"(ts_utc AT TIME ZONE 'America/New_York')::date = :d "
        f"AND lat BETWEEN {LAT_MIN} AND {LAT_MAX} "
        f"AND lon BETWEEN {LON_MIN} AND {LON_MAX} "
        f"AND alt_ft BETWEEN {ALT_MIN_FT} AND {ALT_MAX_FT}"
    )


    sql = text(f"""
        WITH day AS (
          SELECT ts_utc, alt_ft, callsign
          FROM {base}
          WHERE {day_filter}
        ),
        cnt AS (SELECT COUNT(*) AS n FROM day),
        param AS (
          SELECT GREATEST(1, CEIL((SELECT n FROM cnt)::numeric / :target)::int) AS modv
        )
        SELECT d.ts_utc, d.alt_ft
        FROM day d, param p
        WHERE MOD(ABS(hashtext(d.callsign || to_char(date_trunc('minute', d.ts_utc), 'YYYYMMDDHH24MI'))), p.modv) = 0
        ORDER BY d.ts_utc
        LIMIT :lim
    """)

    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"d": day_et, "target": int(target_points), "lim": int(target_points)})

    if len(df) > target_points:
        df = df.sample(target_points, random_state=1).reset_index(drop=True)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_fuel_daily(start_date_et):
    """
    Load per-day fuel and CO2 totals from core.mv_daily_fuel.
    Filters by start_date_et if provided.
    """
    sql = text("""
        SELECT day_et, fuel_kg, co2_kg
        FROM core.mv_daily_fuel
        WHERE day_et >= :d
        ORDER BY day_et
    """)
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"d": start_date_et})
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_heavy_trend(start_date_et):
    """
    Returns one row per ET day with heavy-share stats.
    """
    sql = text(f"""
        WITH base AS (
          SELECT DISTINCT day_et, callsign, icao24
          FROM core.corridor_rows
          WHERE day_et >= :d
        ),
        typed AS (
          SELECT b.day_et, b.callsign,
                 COALESCE(NULLIF(ot.typecode,''), NULLIF(im.typecode,'')) AS typecode
          FROM base b
          LEFT JOIN core.os_aircraft   ot ON ot.icao24 = b.icao24
          LEFT JOIN core.icao_type_map im ON im.icao24 = b.icao24
        ),
        fam AS (
          SELECT t.day_et, t.callsign, fm.family
          FROM typed t
          LEFT JOIN core.typecode_family fm ON fm.typecode = t.typecode
        )
        SELECT
          day_et,
          COUNT(DISTINCT callsign) AS flights_all,
          COUNT(DISTINCT CASE WHEN family IN {HEAVY_FAMILIES} THEN callsign END) AS flights_heavy,
          ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN family IN {HEAVY_FAMILIES} THEN callsign END)
            / NULLIF(COUNT(DISTINCT callsign), 0), 1
          ) AS heavy_share_pct
        FROM fam
        GROUP BY day_et
        ORDER BY day_et
    """)
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"d": start_date_et})
    if not df.empty:
        df["day_et"] = pd.to_datetime(df["day_et"]).dt.date
        df["flights_all"] = df["flights_all"].astype(int)
        df["flights_heavy"] = df["flights_heavy"].astype(int)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_heavy_stack_daily(start_et: str = None, end_et: str = None) -> pd.DataFrame:
    """
    Returns one row per ET day with columns: day_et, heavy, other, total.
    Tries to read from core.mv_heavy_stack_daily if it exists, otherwise falls back to CTE.
    """
    if table_exists("core", "mv_heavy_stack_daily"):
        sql = text("""
            SELECT day_et, heavy, other, total
            FROM core.mv_heavy_stack_daily
            WHERE (:s IS NULL OR day_et >= :s)
              AND (:e IS NULL OR day_et <= :e)
            ORDER BY day_et;
        """)
        with engine.connect() as cn:
            df = pd.read_sql(sql, cn, params={"s": start_et, "e": end_et})
        df["day_et"] = pd.to_datetime(df["day_et"])
        return df

    # Fallback: original CTE (slower)
    sql = text("""
    WITH total AS (
      SELECT day_et, COUNT(DISTINCT callsign) AS total
      FROM core.corridor_rows
      GROUP BY day_et
    ),
    typed AS (
      SELECT
        c.day_et,
        c.callsign,
        COALESCE(NULLIF(ot.typecode,''), NULLIF(im.typecode,'')) AS typecode
      FROM core.corridor_rows c
      LEFT JOIN core.os_aircraft   ot ON ot.icao24 = c.icao24
      LEFT JOIN core.icao_type_map im ON im.icao24 = c.icao24
      GROUP BY c.day_et, c.callsign, ot.typecode, im.typecode
    ),
    fam AS (
      SELECT
        t.day_et,
        t.callsign,
        fm.family
      FROM typed t
      LEFT JOIN core.typecode_family fm ON fm.typecode = t.typecode
    ),
    heavy AS (
      SELECT
        day_et,
        COUNT(DISTINCT callsign) AS heavy
      FROM fam
      WHERE family IN ('B777','B787','A330','A330neo','A350','B75/76')
      GROUP BY day_et
    )
    SELECT
      t.day_et,
      COALESCE(h.heavy,0) AS heavy,
      (t.total - COALESCE(h.heavy,0)) AS other,
      t.total
    FROM total t
    LEFT JOIN heavy h USING(day_et)
    WHERE (:s IS NULL OR t.day_et >= :s)
      AND (:e IS NULL OR t.day_et <= :e)
    ORDER BY t.day_et;
    """)
    with engine.connect() as cn:
        df = pd.read_sql(sql, cn, params={"s": start_et, "e": end_et})
    df["day_et"] = pd.to_datetime(df["day_et"])
    return df

# =========================
# UTILITIES FOR MANEUVERS SUMMARY
# =========================
def _days_available_safe() -> list[str]:
    """Return cached ET-day list if available, otherwise query."""
    try:
        return days_available
    except NameError:
        return list_days_et()


@st.cache_data(ttl=600, show_spinner=False)
def _maneuver_counts_for_day(day_et: str, max_rows: int = 60_000) -> dict:
    """
    Compute simple per-day maneuver counts (unique callsigns):
      - climb / descent / no_level_change
      - s_turn proxy (rolling |Δheading| sum over 5 samples > 180°)
    Now cached per day to avoid recomputing on every rerun.
    """
    df = get_rows_for_maneuvers(day_et, max_rows=max_rows)
    if df.empty:
        return {"day": pd.to_datetime(day_et).date(),
                "climb": 0, "descent": 0, "no_level_change": 0, "s_turn": 0}

    df = df.sort_values(["callsign", "ts_utc"]).reset_index(drop=True)

    # Altitude-based classes
    df["alt_delta"] = df.groupby("callsign", sort=False)["alt_ft"].diff()
    climb_th, desc_th = 30.0, -30.0
    df["climb_desc"] = np.where(
        df["alt_delta"] >= climb_th, "climb",
        np.where(df["alt_delta"] <= desc_th, "descent", "no_level_change")
    )
    agg_cd = (
        df.groupby("climb_desc")["callsign"]
          .nunique()
          .reindex(["climb", "descent", "no_level_change"], fill_value=0)
    )

    # S-turn proxy (needs track_deg)
    s_turn_count = 0
    if "track_deg" in df.columns:
        df["hdg_delta"] = df.groupby("callsign", sort=False)["track_deg"].diff()
        df["hdg_delta"] = ((df["hdg_delta"] + 180) % 360) - 180
        roll_abs_sum = (
            df.groupby("callsign", sort=False)["hdg_delta"]
              .transform(lambda s: s.abs().rolling(5, min_periods=3).sum())
        )
        df["s_turn_bool"] = (roll_abs_sum > 180).fillna(False)
        s_turn_count = df.loc[df["s_turn_bool"], "callsign"].nunique()

    return {
        "day": pd.to_datetime(day_et).date(),
        "climb": int(agg_cd.get("climb", 0)),
        "descent": int(agg_cd.get("descent", 0)),
        "no_level_change": int(agg_cd.get("no_level_change", 0)),
        "s_turn": int(s_turn_count),
    }

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Settings")

# Overview filter: start date (default to 2025-09-25)
start_date_et = st.sidebar.date_input("Start date (ET)", value=_date(2025, 9, 25))

# Sampling controls - lighter defaults for Mac
points_alt = st.sidebar.slider(
    "Max points for Alt vs Time",
    20_000, 200_000, 60_000, 20_000,
    key="pts_alt",
)

points_map = st.sidebar.slider(
    "Max points for Map",
    20_000, 200_000, 200_000, 20_000,   # ← DEFAULT NOW 200,000
    key="pts_map",
)

if st.sidebar.button("Clear cache & rerun"):
    st.cache_data.clear()
    st.rerun()

st.markdown(

    """
<style>
.stTabs { margin-top: 0.8rem; }
.stTabs [data-baseweb="tab"] {
    color: #cfd3dc !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    padding: 12px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    font-weight: 900 !important;
    border-bottom: 4px solid #ff5252 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# TABS
# =========================
tab_overview,tab_daily, tab_map,tab_wind,tab_fuel= st.tabs(
    ["Overview","Daily Pattern Explorer", "Flight Tracks Map", "Wind Effect","Fuel & CO₂"]
)

# ---------- Overview ----------
with tab_overview:
    st.title("Corridor Overview")
    try:
        kpi = get_kpi(start_date_et)
        if kpi.empty:
            st.error("No KPI data. Ensure core.mv_daily_kpis is built/refreshed and start date is correct.")
        else:
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total flights", f"{int(kpi['flights_unique'].sum()):,}")
            c2.metric("Avg / day", f"{kpi['flights_unique'].mean():.1f}")
            c3.metric("Min day", f"{int(kpi['flights_unique'].min()):,}")
            c4.metric("Max day", f"{int(kpi['flights_unique'].max()):,}")

            # Daily flights bar
            fig = px.bar(
                kpi,
                x="day_et",
                y="flights_unique",
                labels={"day_et": "Date (ET)", "flights_unique": "Number of flights"},
                title="Daily Flights (By Callsigns)",
            )
            fig.update_layout(
                xaxis=dict(tickformat="%b %d"),
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            

            # === Maneuvers — last N ET days (lighter for Mac) ===
            st.subheader("Maneuvers")
            _days_src = _days_available_safe()
            if not _days_src:
                st.info("No days available for maneuvers.")
            else:
                # Use fewer days + cached per-day summaries
                N_DAYS = 30
                days_n = _days_src[-N_DAYS:] if len(_days_src) > N_DAYS else _days_src
                with st.spinner(f"Computing per-day maneuver summaries for last {len(days_n)} days…"):
                    rows = [_maneuver_counts_for_day(d, max_rows=60_000) for d in days_n]
                    man = pd.DataFrame(rows).sort_values("day")

                if man.empty:
                    st.info("No maneuver summaries available.")
                else:
                    # Stacked bar: climb / descent / no level change
                    melted = man.melt(
                        id_vars="day",
                        value_vars=["climb", "descent", "no_level_change"],
                        var_name="class", value_name="flights",
                    )
                    class_order = ["climb", "descent", "no_level_change"]
                    label_map = {
                        "climb": "Climb",
                        "descent": "Descent",
                        "no_level_change": "No level change",
                    }
                    melted["class"] = pd.Categorical(
                        melted["class"], categories=class_order, ordered=True
                    )
                    melted["label"] = melted["class"].map(label_map)

                    fig_stack = px.bar(
                        melted,
                        x="day",
                        y="flights",
                        color="label",
                        category_orders={"label": [label_map[c] for c in class_order]},
                        labels={"day": "Date (ET)", "flights": "Number of flights", "label": "Maneuver"},
                        title="Climb / Descent / No level change (unique flights per day)",
                    )
                    fig_stack.update_layout(
                        margin=dict(l=20, r=20, t=50, b=40),
                        barmode="stack",
                    )
                    st.plotly_chart(fig_stack, use_container_width=True)

                    # Separate bar for S-turns (if any flagged)
                    if man["s_turn"].sum() > 0:
                        fig_st = px.bar(
                            man,
                            x="day",
                            y="s_turn",
                            labels={"day": "Date (ET)", "s_turn": "Flights"},
                            title="S-turns",
                        )
                        fig_st.update_layout(margin=dict(l=20, r=20, t=50, b=40))
                        st.plotly_chart(fig_st, use_container_width=True)
                    else:
                        st.caption("S-turn proxy requires heading (`track_deg`). None flagged with current inputs.")

            # Recent flights table for a target ET day (example)
            target_day = str(pd.to_datetime(kpi["day_et"].max()).date())
            st.subheader(f"Most recent 20 flights (ET day {target_day})")
            recent = get_recent_flights_for_day(target_day, limit=20)
            if recent.empty:
                st.info(f"No flights found for {target_day}.")
            else:
                show_cols = ["ts_et", "callsign", "icao24", "lat", "lon", "alt_ft", "groundspeed_kt", "track_deg"]
                present = [c for c in show_cols if c in recent.columns]
                st.dataframe(recent[present], use_container_width=True, height=420)

    except Exception as e:
        st.error(f"Overview error: {e}")

# Precompute available ET days for other tabs
try:
    days_available = list_days_et()
except Exception as e:
    days_available = []
    with tab_overview:
        st.error(f"Failed to list ET days: {e}")

# ---------- Daily Pattern Explorer ----------
with tab_daily:
    st.title("Daily Flight Pattern Explorer")
    if not days_available:
        st.warning("No available ET days.")
    else:
        pick_day = st.selectbox(
            "Select ET day",
            days_available,
            index=len(days_available) - 1,
        )
        try:
            # Hourly counts
            hourly = get_hourly_counts(pick_day)
            if hourly.empty:
                st.warning("No hourly data for this day.")
            else:
                fig_hour = px.bar(
                    hourly,
                    x="hour_utc",
                    y="flights",
                    labels={"hour_utc": "Date/Hour (UTC)", "flights": "Number of Flights"},
                    title=f"Hourly Flights (ET day {pick_day})",
                )
                fig_hour.update_layout(margin=dict(l=20, r=20, t=50, b=40))
                st.plotly_chart(fig_hour, use_container_width=True)

            # Top airlines
            st.subheader("Top 20 Airlines")
            airlines = get_top_airlines(pick_day, limit=20)
            if airlines.empty:
                st.info("No airline breakdown for this day.")
            else:
                fig_air = px.bar(
                    airlines,
                    x="airline",
                    y="flights",
                    labels={"airline": "Airlines", "flights": "Number of flights"},
                    title=f"Top 20 Airlines on {pick_day}",
                )
                fig_air.update_layout(margin=dict(l=20, r=20, t=50, b=40))
                st.plotly_chart(fig_air, use_container_width=True)

            # Altitude vs Time
            st.subheader("Altitude vs Time")
            alt_df = get_alt_time_sample(pick_day, target_points=points_alt)
            if alt_df.empty:
                st.info("No points available for altitude/time.")
            else:
                # 1. Define altitude bins and labels
                bins = [20000, 25000, 30000, 35000, 40000, 45000]
                labels = ["20–25k ft", "25–30k ft", "30–35k ft", "35–40k ft", "40–45k ft"]

                # 2. Create a new column with altitude band
                alt_df["alt_band"] = pd.cut(
                    alt_df["alt_ft"],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                )

                # 3. Scatter plot with color by band
                fig_alt = px.scatter(
                    alt_df,
                    x="ts_utc",
                    y="alt_ft",
                    color="alt_band",
                    labels={
                        "ts_utc": "Time (UTC)",
                        "alt_ft": "Altitude (ft)",
                        "alt_band": "Altitude range",
                    },
                    title="Altitude vs Time (UTC)",
                    opacity=0.6,
                    render_mode="webgl",
                )
                fig_alt.update_traces(marker=dict(size=3))
                fig_alt.update_layout(margin=dict(l=20, r=20, t=50, b=40))
                st.plotly_chart(fig_alt, use_container_width=True)

        except Exception as e:
            st.error(f"Daily Explorer error: {e}")

# ---------- Flight Tracks Map ----------

WAYPOINTS = [
    {"name": "Sea Isle",      "lat": 39.0955089, "lon": -74.8003439},
    {"name": "Armel",         "lat": 38.9345925, "lon": -77.4667017},
    {"name": "Coyle",         "lat": 39.8173381, "lon": -74.4316258},
    {"name": "Robbinsville",  "lat": 40.2024022, "lon": -74.4950261},
    {"name": "Lancaster",     "lat": 40.1199756, "lon": -76.2912953},
    {"name": "Kennedy",       "lat": 40.6328839, "lon": -73.7713917},
]


with tab_map:
    st.title("Corridor Flight Tracks")
    if not days_available:
        st.warning("No available ET days.")
    else:
        c1, c2, c3 = st.columns([1.2, 1.0, 2.2])
        with c1:
            pick_day_map = st.selectbox(
                "Select ET day",
                days_available,
                index=len(days_available) - 1,
                key="map_day",
            )
        with c2:
            dir_choice = st.radio(
                "Direction",
                ["All", "Northbound", "Southbound"],
                horizontal=True,
                key="dir_choice",
            )
        with c3:
            alt_min, alt_max = st.slider(
                "Altitude filter (ft)",
                ALT_MIN_FT,
                ALT_MAX_FT,
                (ALT_MIN_FT, ALT_MAX_FT),
                step=500,
                key="alt_range",
            )

        try:
            sub = get_map_sample(pick_day_map, target_points=points_map)
            if sub.empty:
                st.warning("No data for selected day.")
            else:
                # Guard: if too many points, warn & ask to reduce sliders
                if len(sub) > 200_000:
                    st.warning(
                        f"Returned {len(sub):,} points for map. "
                        "Please lower 'Max points for Map' in the sidebar or narrow filters."
                    )

                # Direction flag per callsign using first/last latitude
                sub = sub.sort_values(["callsign", "ts_utc"])
                grp = sub.groupby("callsign", sort=False)["lat"].agg(first="first", last="last").reset_index()
                grp["dir_flag"] = np.where(
                    grp["last"] >= grp["first"],
                    "Northbound",
                    "Southbound",
                )
                sub = sub.merge(grp[["callsign", "dir_flag"]], on="callsign", how="left")

                # Proper categorical dtype
                cat_dir = CategoricalDtype(categories=["Northbound", "Southbound"], ordered=False)
                sub["dir_flag"] = sub["dir_flag"].astype("string").astype(cat_dir)

                # Apply filters
# Apply filters
            if dir_choice != "All":
             sub = sub[sub["dir_flag"] == dir_choice]

# Basic altitude filter from slider
             sub = sub[sub["alt_ft"].between(alt_min, alt_max)]

# --- Keep only "long" paths in the corridor ---
# 1) Require a minimum number of points per callsign (inside corridor)
             MIN_POINTS_PER_PATH = 50   # you can tweak this (8, 10, 15, …)

             sizes = (
    sub.groupby(["callsign", "dir_flag"])["ts_utc"]
       .transform("size")
)
             sub = sub[sizes >= MIN_POINTS_PER_PATH]

# 2) Optionally also require some span across the box
#    (skip very short zig-zags or tiny segments)
            lon_span = sub.groupby(["callsign", "dir_flag"])["lon"].transform(lambda s: s.max() - s.min())
            lat_span = sub.groupby(["callsign", "dir_flag"])["lat"].transform(lambda s: s.max() - s.min())

            MIN_LON_SPAN = 1.0   # degrees
            MIN_LAT_SPAN = 0.3   # degrees

            sub = sub[(lon_span.abs() >= MIN_LON_SPAN) | (lat_span.abs() >= MIN_LAT_SPAN)]

                # ---- Build line paths per callsign (PathLayer) ----
            span = max(1, (ALT_MAX_FT - ALT_MIN_FT))
            records = []
            for (cs, dflag), g in sub.groupby(["callsign", "dir_flag"], sort=False):
                    if len(g) < 2:
                        continue  # need at least 2 points to draw a line
                    path = g[["lon", "lat"]].to_numpy().tolist()
                    alt_mean = float(g["alt_ft"].mean())
                    # color by mean altitude (blue -> orange)
                    color_v = int(np.clip(((alt_mean - ALT_MIN_FT) / span) * 255, 0, 255))
                    color = [color_v, 128, 255 - color_v, 200]  # RGBA
                    records.append({
                        "callsign": cs,
                        "dir_flag": dflag,
                        "alt_mean": alt_mean,
                        "path": path,
                        "color": color,
                    })

            paths_df = pd.DataFrame.from_records(records)
            if paths_df.empty:
                    st.warning("No paths match the current filters.")
            else:
                    # Layers
                    tile = pdk.Layer(
                        "TileLayer",
                        data="https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png",
                        minZoom=3,
                        maxZoom=18,
                        tile_size=256,
                    )
                    polygon_coords = [[
                        [LON_MIN, LAT_MIN],
                        [LON_MAX, LAT_MIN],
                        [LON_MAX, LAT_MAX],
                        [LON_MIN, LAT_MAX],
                    ]]
                    bbox_layer = pdk.Layer(
                        "PolygonLayer",
                        data=[{"polygon": polygon_coords}],
                        get_polygon="polygon",
                        get_fill_color=[60, 120, 255, 70],
                        get_line_color=[60, 120, 255, 200],
                        line_width_min_pixels=2,
                    )
                    # Waypoints Layer
                    waypoint_layer = pdk.Layer(
                    "ScatterplotLayer",
                     data=WAYPOINTS,
                     get_position=["lon", "lat"],
                     get_color=[255, 255, 0, 200],  # yellow markers
                      get_radius=5000,
                     pickable=True,
                     )

                    tracks = pdk.Layer(
                        "PathLayer",
                        data=paths_df,
                        get_path="path",
                        get_color="color",
                        width_scale=1,
                        width_min_pixels=1,
                        width_max_pixels=6,
                        pickable=True,
                        opacity=0.7,
                    )

                    view = pdk.ViewState(
                        latitude=(LAT_MIN + LAT_MAX) / 2,
                        longitude=(LON_MIN + LON_MAX) / 2,
                        zoom=6.5,
                        pitch=0,
                    )
                    deck = pdk.Deck(
                       layers=[tile, bbox_layer, tracks, waypoint_layer],
                       initial_view_state=view,
                       map_style=None,
tooltip={
    "html": "<b>{name}</b><br/>Lat: {lat}<br/>Lon: {lon}<br/>"
            "<b>{callsign}</b><br/>Dir: {dir_flag}<br/>Mean alt: {alt_mean} ft",
    "style": {"color": "white"}
},

                    )
                    st.pydeck_chart(deck)
                    st.caption(f"{len(paths_df):,} paths")

                    # Legend overlay
                    legend_html = """
<div style="
  position:fixed; bottom:25px; right:25px; z-index:9999;
  padding:10px 14px; background:#1f1f1f; color:#ddd; border:1px solid #2a2a2a;
  border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.35); font-size:14px; line-height:1.25;">
  <div style="font-weight:700;margin-bottom:6px;">Legend</div>
  <div style="display:flex;align-items:center;margin:3px 0;">
    <span style="display:inline-block;width:14px;height:14px;
                 background:rgba(60,120,255,0.25);
                 border:2px solid rgba(60,120,255,0.9);
                 border-radius:2px;margin-right:8px;"></span>
    Corridor Boundary (Baltimore ↔ NYC)
  </div>
  <div style="margin-top:6px;">
    <div style="display:flex;align-items:center;margin-bottom:2px;">
      <span style="margin-right:8px;">Altitude</span>
      <div style="width:140px;height:12px;
                  background:linear-gradient(90deg, rgb(0,128,255), rgb(255,128,0));
                  border-radius:4px;"></div>
      <span style="margin-left:8px;">ft</span>
    </div>
    <div style="display:flex;justify-content:space-between;width:210px;font-size:12px;margin-top:4px;">
           <span>20,000</span><span>43,000</span>
    </div>
  </div>
</div>
"""
                    st.markdown(legend_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Map error: {e}")
# ---------- Wind Effect Tab (inside Overview) ----------
# ---------- Wind Effect Tab ----------
with tab_wind:
    st.title("Wind Effect — Headwind, Crosswind, Direction")

    # Load hourly wind-effect data from corridor_wind_hourly
    try:
        wind_hourly = get_wind_hourly(start_date_et)
    except Exception as e:
        wind_hourly = pd.DataFrame()
        st.error(f"Failed to load hourly wind-effect data: {e}")

    if wind_hourly.empty:
        st.warning(
            "No data in core.corridor_wind_hourly for the selected start date.\n"
            "Make sure the table is populated and start date is not too late."
        )
    else:
        # ---- Basic series ----
        hw = wind_hourly["headwind_kt"]
        cw = wind_hourly["crosswind_kt"]

        # ---- Headwind KPIs (no tailwind metric) ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean headwind (kt)",   f"{hw.mean():.1f}")
        c2.metric("Median headwind (kt)", f"{hw.median():.1f}")
        c3.metric("Max headwind (kt)",    f"{hw.max():.1f}")

        # ==========================
        # Headwind distribution
        # ==========================
        st.subheader("Headwind distribution (hourly corridor averages)")

        fig_hw = px.histogram(
            wind_hourly,
            x="headwind_kt",
            nbins=60,
            labels={"headwind_kt": "Headwind (kt)"},
            title="Histogram of Hourly Corridor Headwind (positive = headwind, negative = tailwind)",
        )
        fig_hw.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_hw.update_layout(margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig_hw, use_container_width=True)

        st.markdown("---")

        # ==========================
        # Crosswind distribution
        # ==========================
        st.subheader("Crosswind distribution (hourly corridor averages)")

        fig_cw = px.histogram(
            wind_hourly,
            x="crosswind_kt",
            nbins=60,
            labels={"crosswind_kt": "Crosswind (kt)"},
            title="Histogram of Hourly Corridor Crosswind",
        )
        fig_cw.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_cw.update_layout(margin=dict(l=20, r=20, t=50, b=40))
        st.plotly_chart(fig_cw, use_container_width=True)

        # Optional: sample table
        with st.expander("Show sample hourly rows"):
            st.dataframe(
                wind_hourly[["hour_et", "headwind_kt", "crosswind_kt", "wind_speed_kt", "avg_gs_kt"]]
                .head(200),
                use_container_width=True,
            )

        st.markdown("---")

        # ================================
        # Wind Direction Distribution
        # ================================
        st.subheader("Wind Direction distribution (hourly corridor averages)")
        st.caption("Histogram of Hourly Corridor Wind Direction (degrees)")

        if "avg_wind_dir_deg" in wind_hourly.columns:
            fig_dir = px.histogram(
                wind_hourly,
                x="avg_wind_dir_deg",
                nbins=36,  # 10-degree bins
                labels={"avg_wind_dir_deg": "Wind Direction (°)"},
                opacity=0.75,
                title="Wind Direction (°)",
            )

            fig_dir.update_layout(
                bargap=0.05,
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(0, 361, 30)),  # 0°, 30°, …, 360°
                    title="Direction (°)",
                ),
                yaxis_title="Count",
                template="plotly_dark",
                showlegend=False,
            )

            st.plotly_chart(fig_dir, use_container_width=True)
        else:
            st.warning("⚠️ Column 'avg_wind_dir_deg' not found in wind_hourly.")

        st.markdown("---")

        # ================================
        # Wind Rose (Direction & Speed)
        # ================================
        st.subheader("Wind Rose (direction & speed)")
        st.caption("Hourly average wind direction and speed in the corridor")

        if (
            not wind_hourly.empty
            and "avg_wind_dir_deg" in wind_hourly.columns
            and "wind_speed_kt" in wind_hourly.columns
        ):
            # Drop NAs just in case
            df_rose = wind_hourly.dropna(
                subset=["avg_wind_dir_deg", "wind_speed_kt"]
            ).copy()

            # --- Direction sectors (16-point compass) ---
            dir_bins = np.arange(-11.25, 371.25, 22.5)  # 16 sectors of 22.5°
            dir_labels = [
                "N", "NNE", "NE", "ENE",
                "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW",
                "W", "WNW", "NW", "NNW",
            ]

            df_rose["dir_sector"] = pd.cut(
                df_rose["avg_wind_dir_deg"] % 360,
                bins=dir_bins,
                labels=dir_labels,
                include_lowest=True,
            )

            # --- Wind speed bins (knots) ---
            speed_bins = [0, 20, 40, 60, 80, 2000]
            speed_labels = ["0–20 kt", "20–40 kt", "40–60 kt", "60–80 kt", "80+ kt"]

            df_rose["speed_bin"] = pd.cut(
                df_rose["wind_speed_kt"],
                bins=speed_bins,
                labels=speed_labels,
                include_lowest=True,
            )

            # Aggregate counts by direction + speed bin
            rose_counts = (
                df_rose.groupby(["dir_sector", "speed_bin"])
                .size()
                .reset_index(name="count")
            )

            # Ensure compass order is correct
            rose_counts["dir_sector"] = rose_counts["dir_sector"].astype("category")
            rose_counts["dir_sector"] = rose_counts["dir_sector"].cat.set_categories(
                dir_labels, ordered=True
            )

            # Plot wind rose
            fig_rose = px.bar_polar(
                rose_counts,
                r="count",
                theta="dir_sector",
                color="speed_bin",
                category_orders={"dir_sector": dir_labels},
                template="plotly_dark",
                labels={
                    "count": "Hours",
                    "dir_sector": "Direction",
                    "speed_bin": "Wind speed (kt)",
                },
            )

            fig_rose.update_layout(
                legend_title_text="Wind speed",
                margin=dict(l=20, r=20, t=40, b=20),
            )

            st.plotly_chart(fig_rose, use_container_width=True)
        else:
            st.info("Wind rose not available: missing `avg_wind_dir_deg` or `wind_speed_kt`.")


# ---------- Fuel & CO₂ Tab ----------
with tab_fuel:
    st.title("Fuel & CO₂")

    # Load daily fuel data from chosen start date
    fuel_df = get_fuel_daily(start_date_et)
    if fuel_df.empty:
        fuel_df = get_fuel_last_n_days(30)
        st.caption("Showing last 30 days because the selected start date returned no data.")

    # Optional: refresh aggregates from SQL with one click


    try:
        if fuel_df.empty:
            st.warning(
                "No fuel/CO₂ data found.\n"
                "Create/refill core.mv_daily_fuel or add fuel_kg/co2_kg to source tables."
            )
        else:
            fuel_df = fuel_df.copy()
            fuel_df["day_et"] = pd.to_datetime(fuel_df["day_et"]).dt.date
            fuel_df["fuel_t"] = fuel_df["fuel_kg"] / 1000.0
            fuel_df["co2_t"] = fuel_df["co2_kg"] / 1000.0

            # Join flights/day to get per-flight fuel (if available)
            have_per_flight = False
            try:
                kpi_df = get_kpi(start_date_et).rename(columns={"day_et": "day_et_dt"})
                kpi_df["day_et"] = pd.to_datetime(kpi_df["day_et_dt"]).dt.date
                fuel_df = fuel_df.merge(
                    kpi_df[["day_et", "flights_unique"]],
                    on="day_et",
                    how="left",
                )
                fuel_df["fuel_kg_per_flight"] = fuel_df["fuel_kg"] / fuel_df["flights_unique"]
                have_per_flight = True
            except Exception:
                pass

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Fuel (t)", f"{fuel_df['fuel_t'].sum():,.1f}")
            c2.metric("Total CO₂ (t)", f"{fuel_df['co2_t'].sum():,.1f}")
            c3.metric("Avg Fuel / day (t)", f"{fuel_df['fuel_t'].mean():,.1f}")
            c4.metric("Avg CO₂ / day (t)", f"{fuel_df['co2_t'].mean():,.1f}")

            # --- Fuel & CO₂ bars (full-width) ---
            df_bars = fuel_df.sort_values("day_et").copy()

            # Fuel
            fig_fuel = px.bar(
                df_bars,
                x="day_et",
                y="fuel_t",
                labels={"fuel_t": "Fuel (tonnes)", "day_et": "Date (ET)"},
                title="Daily Fuel Consumption",
            )
            fig_fuel.update_layout(margin=dict(l=20, r=20, t=50, b=40))
            st.plotly_chart(fig_fuel, use_container_width=True)

            # CO2
            fig_co2 = px.bar(
                df_bars,
                x="day_et",
                y="co2_t",
                labels={"co2_t": "CO₂ (tonnes)", "day_et": "Date (ET)"},
                title="Daily CO₂ Emissions",
            )
            fig_co2.update_layout(margin=dict(l=20, r=20, t=50, b=40))
            st.plotly_chart(fig_co2, use_container_width=True)

                   # --- Top aircraft types over this fuel period ---
            st.markdown("---")
            st.subheader("Top aircraft types (by flights in corridor)")

            start_day = fuel_df["day_et"].min()
            end_day   = fuel_df["day_et"].max()

            top_types = get_top_aircraft_types(start_day, end_day, limit=10)

            if top_types.empty:
                st.caption(
                    "No aircraft-type breakdown available "
                    "(need core.corridor_rows + os_aircraft / icao_type_map)."
                )
            else:
                # nicer label: "TYPE (share%)"
                top_types = top_types.copy()
                top_types["label"] = (  
                    top_types["typecode"]
                    + " (" + top_types["share_pct"].astype(str) + "%)"
                )

                fig_types = px.bar(
                    top_types,
                    x="label",
                    y="flights",
                    labels={
                        "label": "Aircraft type",
                        "flights": "Number of flights",
                    },
                    title=f"Top {len(top_types)} aircraft types "
                          f"({start_day} → {end_day})",
                )
                fig_types.update_layout(
                    xaxis_tickangle=-45,
                    margin=dict(l=20, r=20, t=50, b=80),
                )
                st.plotly_chart(fig_types, use_container_width=True)

                # Optional table under the chart
                with st.expander("Show aircraft-type table"):
                    st.dataframe(
                        top_types[["typecode", "flights", "share_pct"]],
                        use_container_width=True,
                    )
    

    except Exception as e:
        st.error(f"Fuel/CO₂ section error: {e}")

