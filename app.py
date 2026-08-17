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

# Località di riferimento per identificare i punti rilevati (nome -> lat, lon)
REFERENCE_PLACES = {
    "Trento": (46.0679, 11.1211),
    "Bolzano": (46.4983, 11.3548),
    "Rovereto": (45.8905, 11.0404),
    "Riva del Garda": (45.8850, 10.8412),
    "Madonna di Campiglio": (46.2297, 10.8256),
    "San Martino di Castrozza": (46.2612, 11.8014),
    "Passo Rolle": (46.2947, 11.7889),
    "Predazzo": (46.3096, 11.6067),
    "Passo Cereda": (46.1500, 11.8500),
    "Feltre": (46.0164, 11.9078),
    "Belluno": (46.1400, 12.2170),
    "Cortina d'Ampezzo": (46.5405, 12.1357),
    "Agordo": (46.2778, 12.0339),
    "Longarone": (46.2600, 12.3000),
    "Vittorio Veneto": (45.9833, 12.3000),
    "Asiago": (45.8722, 11.5122),
    "Bassano del Grappa": (45.7667, 11.7333),
    "Schio": (45.7167, 11.3500),
    "Verona": (45.4384, 10.9916),
    "Vicenza": (45.5455, 11.5354),
    "Padova": (45.4064, 11.8768),
    "Treviso": (45.6669, 12.2431),
    "Venezia": (45.4408, 12.3155),
    "Rovigo": (45.0705, 11.7905),
    "Arco": (45.9186, 10.8836),
    "Malé": (46.3542, 10.9203),
    "Cavalese": (46.2856, 11.4569),
    "Moena": (46.3733, 11.6567),
    "Canazei": (46.4767, 11.7697),
    "Cortina Marmolada": (46.4300, 11.8500),
}


def decimal_to_dms(value: float, is_lat: bool) -> str:
    """Converte una coordinata decimale in formato DMS, es. 46°31′35″N."""
    hemisphere = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    value = abs(value)
    degrees = int(value)
    minutes_full = (value - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60)
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        degrees += 1
    return f"{degrees}°{minutes:02d}′{seconds:02d}″{hemisphere}"


def coords_to_dms(lat: float, lon: float) -> str:
    return f"{decimal_to_dms(lat, True)} {decimal_to_dms(lon, False)}"


def haversine_km(lat1, lon1, lat2, lon2):
    """Distanza approssimata in km tra due punti geografici."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_place(lat, lon):
    """Ritorna (nome_località, distanza_km) più vicina al punto dato."""
    best_name, best_dist = None, float("inf")
    for name, (plat, plon) in REFERENCE_PLACES.items():
        d = haversine_km(lat, lon, plat, plon)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist


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
            place, dist = nearest_place(lat, lon)
            popup_text = f"{val:.1f} mm — {coords_to_dms(lat, lon)} — vicino a {place} (~{dist:.0f} km)"
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color="#1f4e79",
                fill=True,
                fill_opacity=0.6,
                popup=popup_text,
                tooltip=f"{place}: {val:.1f} mm",
            ).add_to(m)

    folium.LayerControl().add_to(m)

    st.caption(
        f"Ultimo aggiornamento: {data['updated_at']}  |  "
        f"Periodo: ultimi {data['days']} giorni  |  "
        f"Massimo rilevato: {data['vmax']:.1f} mm"
    )

    st_folium(m, width=None, height=850, returned_objects=[])

    with st.expander("📊 Dettaglio punti rilevati"):
        valid_idx = ~np.isnan(data["values"])
        rows = sorted(
            zip(data["lats"][valid_idx], data["lons"][valid_idx], data["values"][valid_idx]),
            key=lambda x: -x[2],
        )
        places_info = [nearest_place(r[0], r[1]) for r in rows]
        st.dataframe(
            {
                "Zona vicina": [p[0] for p in places_info],
                "Dist. dalla zona (km)": [f"{p[1]:.0f}" for p in places_info],
                "Coordinate": [coords_to_dms(r[0], r[1]) for r in rows],
                "Pioggia (mm)": [f"{r[2]:.1f}" for r in rows],
            },
            use_container_width=True,
            hide_index=True,
        )
