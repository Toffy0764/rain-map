"""
Mappa Piogge - Ultimi 15 giorni
--------------------------------
App Streamlit che genera una mappa raster della pioggia cumulata negli
ultimi 15 giorni su un'area personalizzabile (default: Trentino e Veneto).

Fonte dati: Open-Meteo Forecast API (parametro past_days), stesso
approccio usato nell'Indice Fungaiolo per i dati ICON-D2.
"""

import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
from scipy.interpolate import griddata
import matplotlib
from matplotlib.colors import Normalize
from PIL import Image

st.set_page_config(page_title="Mappa Piogge - Ultimi 15 giorni", layout="wide")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Preset area: Trentino-Alto Adige + Veneto (bounding box approssimativo)
PRESETS = {
    "Trentino e Veneto": {"lat_min": 45.4, "lat_max": 47.1, "lon_min": 10.4, "lon_max": 13.0},
    "Solo Trentino": {"lat_min": 45.6, "lat_max": 46.9, "lon_min": 10.4, "lon_max": 12.0},
    "Solo Veneto": {"lat_min": 44.8, "lat_max": 46.8, "lon_min": 10.6, "lon_max": 13.1},
    "Dolomiti (zona ristretta)": {"lat_min": 46.0, "lat_max": 46.7, "lon_min": 11.5, "lon_max": 12.6},
}


# ----------------------------------------------------------------------
# Recupero dati
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_point_rain(lat: float, lon: float, days: int) -> float:
    """Ritorna la pioggia cumulata (mm) sugli ultimi `days` giorni completi."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "past_days": days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        values = data.get("daily", {}).get("precipitation_sum", [])
        # Escludo l'ultimo valore: è il giorno corrente (incompleto)
        completed = values[:-1] if len(values) > days else values
        completed = [v for v in completed if v is not None]
        return float(sum(completed)) if completed else 0.0
    except Exception:
        return float("nan")


def build_grid(lat_min, lat_max, lon_min, lon_max, resolution):
    lats = np.linspace(lat_min, lat_max, resolution)
    lons = np.linspace(lon_min, lon_max, resolution)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    return grid_lat.flatten(), grid_lon.flatten()


def fetch_all_points(lats, lons, days, max_workers=12):
    values = np.full(len(lats), np.nan)
    progress = st.progress(0.0, text="Recupero dati pioggia da Open-Meteo...")
    total = len(lats)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(fetch_point_rain, lat, lon, days): i
            for i, (lat, lon) in enumerate(zip(lats, lons))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            values[idx] = future.result()
            done += 1
            progress.progress(done / total, text=f"Recupero dati pioggia... {done}/{total}")

    progress.empty()
    return values


# ----------------------------------------------------------------------
# Interpolazione e raster
# ----------------------------------------------------------------------

def interpolate_raster(lats, lons, values, lat_min, lat_max, lon_min, lon_max, raster_res=300):
    valid = ~np.isnan(values)
    if valid.sum() < 4:
        return None

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, raster_res),
        np.linspace(lat_min, lat_max, raster_res),
    )

    points = np.column_stack([lons[valid], lats[valid]])

    grid_z = griddata(points, values[valid], (grid_lon, grid_lat), method="cubic")
    # Riempio i buchi (NaN da estrapolazione cubic) con interpolazione lineare
    mask_nan = np.isnan(grid_z)
    if mask_nan.any():
        grid_z_lin = griddata(points, values[valid], (grid_lon, grid_lat), method="nearest")
        grid_z[mask_nan] = grid_z_lin[mask_nan]

    grid_z = np.clip(grid_z, 0, None)
    return grid_z


def raster_to_png_overlay(grid_z, vmax=None):
    """Converte la matrice di pioggia in un'immagine PNG RGBA (scala di blu)."""
    if vmax is None:
        vmax = np.nanmax(grid_z) if np.nanmax(grid_z) > 0 else 1.0

    norm = Normalize(vmin=0, vmax=vmax)
    cmap = matplotlib.colormaps["Blues"]

    rgba = cmap(norm(grid_z))
    rgba[..., 3] = 0.75  # trasparenza costante per sovrapposizione su mappa

    # Origine in basso a sinistra -> l'array va capovolto per l'immagine
    rgba_flipped = np.flipud(rgba)
    img = Image.fromarray((rgba_flipped * 255).astype(np.uint8), mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}", vmax


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🌧️ Mappa Piogge - Ultimi 15 giorni")
st.caption("Precipitazione cumulata su area personalizzabile (fonte: Open-Meteo)")

with st.sidebar:
    st.header("Impostazioni area")

    preset_choice = st.selectbox("Area predefinita", list(PRESETS.keys()) + ["Personalizzata"])

    if preset_choice == "Personalizzata":
        col1, col2 = st.columns(2)
        with col1:
            lat_min = st.number_input("Lat minima", value=45.4, format="%.2f")
            lon_min = st.number_input("Lon minima", value=10.4, format="%.2f")
        with col2:
            lat_max = st.number_input("Lat massima", value=47.1, format="%.2f")
            lon_max = st.number_input("Lon massima", value=13.0, format="%.2f")
    else:
        preset = PRESETS[preset_choice]
        lat_min, lat_max = preset["lat_min"], preset["lat_max"]
        lon_min, lon_max = preset["lon_min"], preset["lon_max"]
        st.info(f"Lat: {lat_min}–{lat_max}  |  Lon: {lon_min}–{lon_max}")

    st.divider()
    st.header("Impostazioni dati")

    days = st.slider("Giorni da considerare", min_value=3, max_value=15, value=15)
    resolution = st.slider(
        "Densità griglia di rilevamento",
        min_value=5, max_value=18, value=10,
        help="Punti per lato interrogati su Open-Meteo. Più alto = più dettaglio ma più chiamate API e tempo di attesa.",
    )
    raster_res = st.slider(
        "Risoluzione mappa raster (pixel)",
        min_value=100, max_value=500, value=300, step=50,
        help="Densità dell'immagine interpolata finale (indipendente dal numero di chiamate API).",
    )

    n_points = resolution * resolution
    st.caption(f"Punti da interrogare: {n_points}")
    if n_points > 250:
        st.warning("Griglia molto fitta: il caricamento potrebbe richiedere diversi minuti.")

    run = st.button("🔄 Genera mappa", type="primary", use_container_width=True)


if "raster_data" not in st.session_state:
    st.session_state.raster_data = None

if run:
    lats, lons = build_grid(lat_min, lat_max, lon_min, lon_max, resolution)
    values = fetch_all_points(lats, lons, days)

    n_valid = np.sum(~np.isnan(values))
    if n_valid < 4:
        st.error("Non è stato possibile recuperare abbastanza dati. Riprova o riduci la griglia.")
    else:
        grid_z = interpolate_raster(lats, lons, values, lat_min, lat_max, lon_min, lon_max, raster_res)
        overlay_url, vmax = raster_to_png_overlay(grid_z)

        st.session_state.raster_data = {
            "overlay_url": overlay_url,
            "vmax": vmax,
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
            "lats": lats, "lons": lons, "values": values,
            "updated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "days": days,
        }

data = st.session_state.raster_data

if data is None:
    st.info("Imposta l'area e premi **Genera mappa** per iniziare.")
else:
    center_lat = (data["bounds"][0][0] + data["bounds"][1][0]) / 2
    center_lon = (data["bounds"][0][1] + data["bounds"][1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")

    folium.raster_layers.ImageOverlay(
        image=data["overlay_url"],
        bounds=data["bounds"],
        opacity=0.75,
        interactive=False,
        cross_origin=False,
    ).add_to(m)

    # Marker sui punti effettivamente rilevati, per riferimento
    for lat, lon, val in zip(data["lats"], data["lons"], data["values"]):
        if not np.isnan(val):
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color="#1f4e79",
                fill=True,
                fill_opacity=0.6,
                popup=f"{val:.1f} mm",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    st.caption(
        f"Ultimo aggiornamento: {data['updated_at']}  |  "
        f"Periodo: ultimi {data['days']} giorni  |  "
        f"Massimo rilevato: {data['vmax']:.1f} mm"
    )

    st_folium(m, width=None, height=650, returned_objects=[])

    with st.expander("📊 Dettaglio punti rilevati"):
        valid_idx = ~np.isnan(data["values"])
        rows = sorted(
            zip(data["lats"][valid_idx], data["lons"][valid_idx], data["values"][valid_idx]),
            key=lambda x: -x[2],
        )
        st.dataframe(
            {
                "Lat": [f"{r[0]:.3f}" for r in rows],
                "Lon": [f"{r[1]:.3f}" for r in rows],
                "Pioggia (mm)": [f"{r[2]:.1f}" for r in rows],
            },
            use_container_width=True,
            hide_index=True,
        )
