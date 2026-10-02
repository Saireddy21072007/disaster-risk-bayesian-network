"""
Standalone Folium map export.

The live dashboard draws its own Leaflet layer, but for the report and for
sharing a snapshot over WhatsApp it is handy to have a single self-contained HTML
file. This writes artifacts/risk_map_<scenario>.html.

Run:  python -m src.gis monsoon_depression

Owner: Jithin Reddy K
"""

from __future__ import annotations

import sys

import folium

from .config import ARTIFACT_DIR, risk_band
from .decision import recommend
from .discretize import evidence_from_reading
from .engine import get_engine
from .scenarios import SCENARIOS, build


def build_map(scenario: str = "monsoon_depression"):
    engine = get_engine()
    readings = build(scenario)

    m = folium.Map(location=[12.4, 78.5], zoom_start=6,
                   tiles="OpenStreetMap", control_scale=True)

    title = (f"<div style='position:fixed;top:10px;left:60px;z-index:9999;"
             f"background:white;padding:6px 12px;border-radius:6px;"
             f"font:600 14px sans-serif;box-shadow:0 1px 4px #0003'>"
             f"{SCENARIOS[scenario]}</div>")
    m.get_root().html.add_child(folium.Element(title))

    for r in readings:
        evidence = evidence_from_reading(r)
        hazards = engine.hazard_probabilities(evidence)
        plan = recommend(engine, evidence, population=r["population"])
        top = plan["dominant_hazard"]
        band, colour = risk_band(hazards[top])

        rows = "".join(
            f"<tr><td>{h}</td><td style='text-align:right'>{p:.0%}</td></tr>"
            for h, p in sorted(hazards.items(), key=lambda kv: -kv[1]))
        gauge = ("<div style='color:#b26a00'>river gauge not reporting</div>"
                 if r.get("river_level_frac") is None else "")

        popup = folium.Popup(f"""
            <div style="font:13px/1.5 sans-serif;min-width:250px">
              <b style="font-size:15px">{r['district']}</b>, {r['state']}<br>
              {gauge}
              <table style="width:100%;margin:6px 0">{rows}</table>
              <b>Action:</b> {plan.get('action_label', plan['action'])}<br>
              <b>Rescue / hospital surge:</b> {plan['p_emergency']:.0%}<br>
              <b>Shelter:</b> {r['nearest_shelter']}<br>
              <span style="color:#666">rain {r['rainfall_mm']} mm &middot;
              {r['temperature_c']} C &middot; wind {r['wind_kmph']} kmph</span>
            </div>""", max_width=320)

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=8 + 16 * hazards[top],
            color=colour,
            weight=3 if r.get("river_level_frac") is None else 1.5,
            fill=True, fill_color=colour,
            fill_opacity=0.12 if r.get("river_level_frac") is None else 0.55,
            popup=popup,
            tooltip=f"{r['district']}: {top} {hazards[top]:.0%} ({band})",
        ).add_to(m)

    legend = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
                padding:8px 12px;border-radius:6px;font:12px sans-serif;
                box-shadow:0 1px 4px #0003">
      <b>Top-hazard probability</b><br>
      <span style="color:#d7301f">&#9679;</span> High (&ge;75%)&nbsp;
      <span style="color:#fdae61">&#9679;</span> Medium&nbsp;
      <span style="color:#1a9850">&#9679;</span> Low<br>
      hollow ring = a sensor is offline
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    out = ARTIFACT_DIR / f"risk_map_{scenario}.html"
    m.save(str(out))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "monsoon_depression"
    print("wrote", build_map(which))
