"""
Live sensor ingestion. This module is the answer to the first half of the
zeroth-review feedback: "it is mentioned real-time sensor data, but you are
choosing the existing preprocessed data?"

Everything below is a genuine live pull, no API key, three separate public
services:

  weather  api.open-meteo.com          rainfall, temperature, humidity, wind,
                                       surface pressure, reanalysis soil moisture
  river    flood-api.open-meteo.com    GloFAS daily river discharge, m3/s,
                                       plus six years of history for calibration
  sea      marine-api.open-meteo.com   sea surface temperature offshore

Three things here are less obvious than "call an API", and each is a modelling
decision we should be able to defend:

1. AGGREGATION TO THE VARIABLE WE ACTUALLY MODEL.
   Our Rainfall node is an IMD 24-hour band, not an instantaneous rate, so we sum
   the trailing 24 hourly values. Temperature is a 24-hour MAX (the heat-wave
   criterion is about the daily maximum), humidity is a 24-hour mean, wind is a
   24-hour max, and PressureDrop is p(t-24h) - p(t). Reading the "current" field
   for these would silently model a different quantity.

2. THREE INDEPENDENT FORECAST MODELS, NOT ONE.
   We request precipitation, temperature and wind from GFS, ECMWF-IFS and ICON
   separately. They disagree, and that disagreement is real information: the
   spread becomes a reliability estimate in measurement.py instead of being
   averaged away.

3. THE RIVER THRESHOLD IS CALIBRATED, NOT INVENTED.
   GloFAS gives discharge in m3/s; our RiverLevel node is a fraction of the local
   danger level. We do not have the CWC gazetted danger level per district, so we
   compute the median annual maximum discharge over six years - the 2-year return
   period, the standard bankfull proxy in hydrology - and express today's
   discharge as a fraction of that. It is cached on disk because it only needs
   recomputing once a season.

Failure is a first-class outcome. Every fetch is wrapped; a service that is down
produces an absent sensor, not an exception, and an absent sensor is exactly what
the Bayesian network already knows how to handle.

Owner: Sai Vandith (adapters) and Sai Reddy A (aggregation + calibration)
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from .config import ARTIFACT_DIR, DISTRICTS_CSV

IST = timezone(timedelta(hours=5, minutes=30))

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# the three numerical weather models we treat as independent sources
NWP_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
SOURCE_LABEL = {
    "gfs_seamless": "GFS (NOAA)",
    "ecmwf_ifs025": "ECMWF IFS",
    "icon_seamless": "ICON (DWD)",
    "glofas": "GloFAS river discharge",
    "marine": "Marine SST",
    "reanalysis": "ERA5-land reanalysis",
}

CACHE_DIR = ARTIFACT_DIR / "feed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = ARTIFACT_DIR / "river_calibration.json"
SNAPSHOT_FILE = ARTIFACT_DIR / "live_snapshot.json"

LIVE_TTL_SECONDS = 15 * 60          # the upstream models update every 15-60 min
CALIBRATION_TTL_SECONDS = 30 * 24 * 3600

# soil porosity used to turn volumetric water content (m3/m3) into the
# saturation fraction our SoilMoisture bins are defined on. 0.45 is a typical
# value for the loamy soils of the Western Ghats; it is an approximation and the
# measurement layer gives this sensor a correspondingly low reliability.
SOIL_POROSITY = 0.45


# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """One number, from one source, about one node, at one time."""
    node: str                  # Bayesian-network node, e.g. "Rainfall"
    column: str                # raw column name, e.g. "rainfall_mm"
    value: float
    source: str                # key into SOURCE_LABEL
    observed_at: str           # ISO timestamp the reading refers to
    fetched_at: str            # ISO timestamp we pulled it
    unit: str = ""
    note: str = ""

    def age_minutes(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(IST)
        try:
            ts = datetime.fromisoformat(self.observed_at)
        except ValueError:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        return max((now - ts).total_seconds() / 60.0, 0.0)


@dataclass
class DistrictFeed:
    """Everything we managed to collect for one district on one pass."""
    district: str
    observations: List[Observation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fetched_at: str = ""
    from_cache: bool = False

    def by_node(self) -> Dict[str, List[Observation]]:
        out: Dict[str, List[Observation]] = {}
        for o in self.observations:
            out.setdefault(o.node, []).append(o)
        return out

    def to_dict(self) -> dict:
        return {
            "district": self.district,
            "fetched_at": self.fetched_at,
            "from_cache": self.from_cache,
            "errors": self.errors,
            "observations": [asdict(o) for o in self.observations],
        }

    @staticmethod
    def from_dict(blob: dict) -> "DistrictFeed":
        f = DistrictFeed(district=blob["district"],
                         errors=list(blob.get("errors", [])),
                         fetched_at=blob.get("fetched_at", ""),
                         from_cache=True)
        f.observations = [Observation(**o) for o in blob.get("observations", [])]
        return f


# ---------------------------------------------------------------------------
def districts() -> pd.DataFrame:
    return pd.read_csv(DISTRICTS_CSV)


def _get(url: str, params: dict, timeout: float = 20.0,
         retries: int = 2) -> dict:
    """GET with a small backoff. Raises on final failure; callers catch."""
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:               # noqa: BLE001 - we re-raise below
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url} failed after {retries + 1} tries: {last}")


def _cache_path(kind: str, lat: float, lon: float) -> Path:
    return CACHE_DIR / f"{kind}_{lat:.3f}_{lon:.3f}.json"


def _cached_get(kind: str, url: str, params: dict, lat: float, lon: float,
                ttl: float) -> tuple[dict, bool]:
    """Returns (payload, came_from_cache). Falls back to a stale cache on error."""
    path = _cache_path(kind, lat, lon)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        return json.loads(path.read_text(encoding="utf-8")), True
    try:
        payload = _get(url, params)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload, False
    except Exception:
        if path.exists():
            # a stale cache beats no reading; the age shows up in staleness decay
            return json.loads(path.read_text(encoding="utf-8")), True
        raise


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------
def _trailing(times: List[str], values: List[Optional[float]], hours: int,
              now: datetime) -> tuple[List[float], Optional[str]]:
    """The last `hours` non-null values at or before now, plus that timestamp."""
    keep, last_ts = [], None
    for t, v in zip(times, values):
        if v is None:
            continue
        ts = datetime.fromisoformat(t).replace(tzinfo=IST)
        if ts <= now:
            keep.append((ts, float(v)))
    if not keep:
        return [], None
    keep.sort()
    cutoff = keep[-1][0] - timedelta(hours=hours - 1)
    window = [v for ts, v in keep if ts >= cutoff]
    last_ts = keep[-1][0].isoformat()
    return window, last_ts


# ---------------------------------------------------------------------------
# the live feed
# ---------------------------------------------------------------------------
class OpenMeteoFeed:
    """
    Live adapter. `offline=True` makes every call use whatever is in the cache
    and never touch the network, which is how the test suite runs.
    """

    def __init__(self, offline: bool = False, ttl: float = LIVE_TTL_SECONDS):
        self.offline = offline
        self.ttl = math.inf if offline else ttl

    # ------------------------------------------------------------- weather
    def weather(self, lat: float, lon: float, now: datetime) -> tuple[List[Observation], List[str]]:
        obs: List[Observation] = []
        errors: List[str] = []
        fetched = now.isoformat()

        hourly = ["precipitation", "temperature_2m", "relative_humidity_2m",
                  "wind_speed_10m", "surface_pressure"]
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(hourly),
            "models": ",".join(NWP_MODELS),
            "past_days": 2, "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        try:
            payload, cached = _cached_get("weather", WEATHER_URL, params,
                                          lat, lon, self.ttl)
        except Exception as exc:
            return [], [f"weather feed unavailable: {exc}"]

        h = payload.get("hourly", {})
        times = h.get("time", [])

        for model in NWP_MODELS:
            # 24-hour rainfall accumulation: the quantity the IMD bands describe
            rain, ts = _trailing(times, h.get(f"precipitation_{model}", []), 24, now)
            if rain and ts:
                obs.append(Observation("Rainfall", "rainfall_mm", round(sum(rain), 2),
                                       model, ts, fetched, "mm/24h",
                                       "sum of trailing 24 hourly values"))
            # heat-wave criteria are about the daily maximum
            temp, ts = _trailing(times, h.get(f"temperature_2m_{model}", []), 24, now)
            if temp and ts:
                obs.append(Observation("Temperature", "temperature_c",
                                       round(max(temp), 2), model, ts, fetched,
                                       "deg C", "24-hour maximum"))
            hum, ts = _trailing(times, h.get(f"relative_humidity_2m_{model}", []), 24, now)
            if hum and ts:
                obs.append(Observation("Humidity", "humidity_pct",
                                       round(sum(hum) / len(hum), 1), model, ts,
                                       fetched, "%", "24-hour mean"))
            wind, ts = _trailing(times, h.get(f"wind_speed_10m_{model}", []), 24, now)
            if wind and ts:
                obs.append(Observation("WindSpeed", "wind_kmph", round(max(wind), 1),
                                       model, ts, fetched, "km/h", "24-hour maximum"))
            # pressure DROP over 24 h, not the pressure itself
            pres, ts = _trailing(times, h.get(f"surface_pressure_{model}", []), 25, now)
            if len(pres) >= 25 and ts:
                drop = pres[0] - pres[-1]
                obs.append(Observation("PressureDrop", "pressure_drop_hpa",
                                       round(max(drop, 0.0), 2), model, ts, fetched,
                                       "hPa/24h", "p(t-24h) - p(t)"))

        if cached and not self.offline:
            errors.append("weather served from cache")

        # NO SOIL-MOISTURE FEED - and this was a deliberate reversal.
        #
        # We did wire one up, from the forecast API's soil_moisture_9_to_27cm. Then
        # we went to relearn the CPTs from the archive API and found that its
        # soil_moisture_7_to_28cm_mean is a different product on a different scale:
        # for the same Coimbatore grid cell the forecast said 0.153 m3/m3 while the
        # archive said 0.33 for a comparable day. Training on one and inferring from
        # the other would have been a silent bug, and inventing a rescaling factor to
        # paper over it would have been worse.
        #
        # So soil moisture stays latent and the temporal filter infers it from the
        # rainfall history, which is also what a district with no soil probes would
        # have to do. See docs/realtime_pipeline.md.
        return obs, errors

    # --------------------------------------------------------------- river
    def river_calibration(self, lat: float, lon: float) -> Optional[dict]:
        """
        Median annual maximum discharge over six years - the 2-year return period,
        the conventional bankfull proxy. Cached for a month.
        """
        key = f"{lat:.3f}_{lon:.3f}"
        store = {}
        if CALIBRATION_FILE.exists():
            store = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            entry = store.get(key)
            if entry and (time.time() - entry.get("computed_at", 0)) < CALIBRATION_TTL_SECONDS:
                return entry
        if self.offline:
            return store.get(key)

        end = datetime.now(IST).date().replace(month=12, day=31) - timedelta(days=365)
        start = end.replace(year=end.year - 5, month=1, day=1)
        try:
            payload = _get(FLOOD_URL, {
                "latitude": lat, "longitude": lon, "daily": "river_discharge",
                "start_date": start.isoformat(), "end_date": end.isoformat(),
            }, timeout=40.0)
        except Exception:
            return store.get(key)

        daily = payload.get("daily", {})
        raw = daily.get("river_discharge", [])
        pairs = [(t, v) for t, v in zip(daily.get("time", []), raw)
                 if v is not None]

        if raw and not pairs:
            # Every value is null: GloFAS models river reaches on a grid, and some
            # of our districts (Chennai, for one) simply have no modelled reach at
            # their centroid. That is a real gap in the data, not an outage, and we
            # record it as such so the dashboard can say so honestly.
            entry = {"bankfull_discharge": None, "years": 0,
                     "method": "no GloFAS river reach at this location",
                     "computed_at": time.time()}
            store[key] = entry
            CALIBRATION_FILE.write_text(json.dumps(store, indent=1), encoding="utf-8")
            return entry

        if len(pairs) < 365:
            return store.get(key)

        by_year: Dict[str, List[float]] = {}
        for t, v in pairs:
            by_year.setdefault(t[:4], []).append(float(v))
        annual_max = [max(vs) for vs in by_year.values() if len(vs) > 300]
        if not annual_max:
            return store.get(key)

        entry = {
            "bankfull_discharge": round(statistics.median(annual_max), 2),
            "annual_maxima": [round(a, 1) for a in sorted(annual_max)],
            "years": len(annual_max),
            "window": f"{start.isoformat()}..{end.isoformat()}",
            "method": "median annual maximum (2-year return period)",
            "computed_at": time.time(),
        }
        store[key] = entry
        CALIBRATION_FILE.write_text(json.dumps(store, indent=1), encoding="utf-8")
        return entry

    def river(self, lat: float, lon: float, now: datetime) -> tuple[List[Observation], List[str]]:
        cal = self.river_calibration(lat, lon)
        if cal and cal.get("bankfull_discharge") is None:
            return [], ["no GloFAS river reach at this district centroid - "
                        "RiverLevel is genuinely unobservable here"]
        if not cal or not cal.get("bankfull_discharge"):
            return [], ["river gauge unavailable: no discharge calibration"]

        try:
            payload, cached = _cached_get("flood", FLOOD_URL, {
                "latitude": lat, "longitude": lon,
                "daily": "river_discharge", "past_days": 7, "forecast_days": 1,
            }, lat, lon, self.ttl)
        except Exception as exc:
            return [], [f"river gauge unavailable: {exc}"]

        daily = payload.get("daily", {})
        pairs = [(t, v) for t, v in zip(daily.get("time", []),
                                        daily.get("river_discharge", []))
                 if v is not None]
        if not pairs:
            return [], ["river gauge returned no discharge"]

        ts, q = pairs[-1]
        frac = float(q) / cal["bankfull_discharge"]
        errors = ["river served from cache"] if (cached and not self.offline) else []
        return [Observation("RiverLevel", "river_level_frac", round(frac, 3),
                            "glofas", f"{ts}T00:00:00", now.isoformat(),
                            "fraction of bankfull",
                            f"{q} m3/s / bankfull {cal['bankfull_discharge']} m3/s")], errors

    # ----------------------------------------------------------------- sea
    def sst(self, lat: float, lon: float, coast_side: str,
            now: datetime) -> tuple[List[Observation], List[str]]:
        """Only meaningful for a coastal district; the point is nudged offshore."""
        if coast_side == "west":
            plat, plon = lat, lon - 0.7
        elif coast_side == "east":
            plat, plon = lat, lon + 0.7
        else:
            return [], []          # inland: the sensor is genuinely absent

        try:
            payload, cached = _cached_get("marine", MARINE_URL, {
                "latitude": plat, "longitude": plon,
                "current": "sea_surface_temperature",
                "timezone": "Asia/Kolkata",
            }, plat, plon, self.ttl)
        except Exception as exc:
            return [], [f"sea surface temperature unavailable: {exc}"]

        cur = payload.get("current", {})
        val = cur.get("sea_surface_temperature")
        if val is None:
            return [], ["marine feed returned no sea surface temperature"]
        errors = ["marine served from cache"] if (cached and not self.offline) else []
        return [Observation("SeaSurfaceTemp", "sst_c", float(val), "marine",
                            cur.get("time", now.isoformat()), now.isoformat(),
                            "deg C", f"offshore point {plat:.2f},{plon:.2f}")], errors

    # ------------------------------------------------------------- one pass
    def fetch_district(self, row, now: Optional[datetime] = None) -> DistrictFeed:
        now = now or datetime.now(IST)
        feed = DistrictFeed(district=row["district"], fetched_at=now.isoformat())

        for getter in (
            lambda: self.weather(float(row["lat"]), float(row["lon"]), now),
            lambda: self.river(float(row["lat"]), float(row["lon"]), now),
            lambda: self.sst(float(row["lat"]), float(row["lon"]),
                             str(row["coast_side"]), now),
        ):
            try:
                obs, errs = getter()
            except Exception as exc:                       # noqa: BLE001
                obs, errs = [], [f"feed error: {exc}"]
            feed.observations.extend(obs)
            feed.errors.extend(errs)

        feed.from_cache = any("cache" in e for e in feed.errors)
        return feed

    def fetch_all(self, only: Optional[Iterable[str]] = None,
                  now: Optional[datetime] = None) -> List[DistrictFeed]:
        df = districts()
        if only:
            wanted = {d.lower() for d in only}
            df = df[df["district"].str.lower().isin(wanted)]
        return [self.fetch_district(row, now) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# snapshots, so a demo is reproducible and works with the wifi off
# ---------------------------------------------------------------------------
def save_snapshot(feeds: List[DistrictFeed], path: Path = SNAPSHOT_FILE) -> Path:
    path.write_text(json.dumps({
        "saved_at": datetime.now(IST).isoformat(),
        "feeds": [f.to_dict() for f in feeds],
    }, indent=1), encoding="utf-8")
    return path


def load_snapshot(path: Path = SNAPSHOT_FILE) -> List[DistrictFeed]:
    if not path.exists():
        return []
    blob = json.loads(path.read_text(encoding="utf-8"))
    return [DistrictFeed.from_dict(f) for f in blob.get("feeds", [])]


class SnapshotFeed:
    """
    Replays a saved snapshot. Same interface as OpenMeteoFeed.

    It re-reads the file whenever the file changes on disk. The first version
    loaded once in __init__, and because the API keeps one processor alive for the
    life of the process, taking a fresh snapshot did nothing until the server was
    restarted - which cost us a confusing ten minutes staring at a sensor we had
    already removed from the pipeline.
    """

    def __init__(self, path: Path = SNAPSHOT_FILE):
        self.path = path
        self._mtime: float = -1.0
        self._feeds: Dict[str, DistrictFeed] = {}
        self._refresh()

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self._feeds = {f.district: f for f in load_snapshot(self.path)}
            self._mtime = mtime

    def fetch_district(self, row, now=None) -> DistrictFeed:
        self._refresh()
        feed = self._feeds.get(row["district"])
        if feed is None:
            return DistrictFeed(district=row["district"],
                                errors=["no snapshot for this district"],
                                fetched_at=(now or datetime.now(IST)).isoformat())
        return feed

    def fetch_all(self, only=None, now=None) -> List[DistrictFeed]:
        df = districts()
        if only:
            wanted = {d.lower() for d in only}
            df = df[df["district"].str.lower().isin(wanted)]
        return [self.fetch_district(row, now) for _, row in df.iterrows()]


def get_feed(mode: str = "live"):
    """mode: live | offline | snapshot"""
    if mode == "snapshot":
        return SnapshotFeed()
    return OpenMeteoFeed(offline=(mode == "offline"))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["Idukki", "Chennai", "Madurai"]
    feed = OpenMeteoFeed()
    feeds = feed.fetch_all(only=which)
    for f in feeds:
        print(f"\n=== {f.district} ({len(f.observations)} readings) ===")
        for o in f.observations:
            print(f"  {o.node:<15} {o.value:>8}  {o.unit:<18} "
                  f"{SOURCE_LABEL.get(o.source, o.source):<24} "
                  f"age {o.age_minutes():.0f} min")
        for e in f.errors:
            print(f"  ! {e}")
    print(f"\nsnapshot -> {save_snapshot(feeds)}")
