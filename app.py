import streamlit as st
import pandas as pd
import random
import math
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Green Route Planner", layout="wide")

st.title("🌿 Green Route Planner")
st.caption("Stable Final Year Project Version")

# ---------------------------------
# GEOCODER
# ---------------------------------
geo = Nominatim(user_agent="green_route_project")

def get_location(place):
    try:
        loc = geo.geocode(place, timeout=10)
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        return None
    return None

# ---------------------------------
# DISTANCE CALCULATION
# ---------------------------------
def distance_km(a, b):
    R = 6371
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))
    return R * c

# ---------------------------------
# TRAIN ML MODEL
# ---------------------------------
@st.cache_resource
def train_model():
    rows = []

    for _ in range(400):
        dist = random.uniform(2, 30)
        traffic = random.randint(1, 10)
        stops = random.randint(0, 8)

        fuel = dist * 0.08 + traffic * 0.03 + stops * 0.02

        rows.append([dist, traffic, stops, fuel])

    df = pd.DataFrame(rows, columns=["dist","traffic","stops","fuel"])

    X = df[["dist","traffic","stops"]]
    y = df["fuel"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)

    return model

model = train_model()

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("Enter Route Details")

source = st.sidebar.text_input("Source", "Mysore Palace")
destination = st.sidebar.text_input("Destination", "Infosys Mysore")

run = st.sidebar.button("Find Routes")

# ---------------------------------
# MAIN
# ---------------------------------
if run:

    src = get_location(source)
    dst = get_location(destination)

    if not src or not dst:
        st.error("Invalid location names.")
        st.stop()

    base_distance = round(distance_km(src, dst), 2)

    routes = []

    for i in range(3):

        extra = random.uniform(0, 3)
        km = round(base_distance + extra, 2)

        traffic = random.randint(2, 9)
        stops = random.randint(0, 7)
        greenery = random.randint(40, 95)

        fuel = model.predict([[km, traffic, stops]])[0]
        fuel = round(fuel, 2)

        time = round((km / 35) * 60 + traffic * 3, 1)

        eco = round(
            (100 - traffic * 5)
            + greenery * 0.5
            - fuel * 8
            - km,
            2
        )

        routes.append({
            "Route": f"Route {i+1}",
            "Distance (km)": km,
            "Time (min)": time,
            "Traffic": traffic,
            "Fuel (L)": fuel,
            "Greenery": greenery,
            "Eco Score": eco
        })

    df = pd.DataFrame(routes)

    fast = df["Time (min)"].idxmin()
    short = df["Distance (km)"].idxmin()
    eco = df["Eco Score"].idxmax()

    c1,c2,c3 = st.columns(3)

    c1.success(f"⚡ Fastest: {df.loc[fast,'Route']}")
    c2.info(f"📏 Shortest: {df.loc[short,'Route']}")
    c3.success(f"🌿 Eco Best: {df.loc[eco,'Route']}")

    st.subheader("Route Comparison")
    st.dataframe(df, use_container_width=True)

    # MAP
    center = [(src[0]+dst[0])/2, (src[1]+dst[1])/2]

    fmap = folium.Map(location=center, zoom_start=12)

    folium.Marker(src, tooltip="Source").add_to(fmap)
    folium.Marker(dst, tooltip="Destination").add_to(fmap)

    colors = ["blue","red","green"]

    for i in range(3):
        mid = [
            center[0] + random.uniform(-0.02,0.02),
            center[1] + random.uniform(-0.02,0.02)
        ]

        points = [src, mid, dst]

        folium.PolyLine(
            points,
            color=colors[i],
            weight=5,
            tooltip=f"Route {i+1}"
        ).add_to(fmap)

    st.subheader("Map View")
    st_folium(fmap, width=1200, height=600)

else:
    st.info("Enter source and destination then click Find Routes.")