import streamlit as st
import pandas as pd
import random
import math
import numpy as np
import heapq
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Green Planner", layout="wide")

st.title("🌿 Smart Green Planning System")
st.caption("Route Optimization + Road Construction Planning")

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.selectbox(
    "Select Module",
    ["Route Planner", "Road Construction Planner"]
)

# =================================================
# ================= MODULE 1 =======================
# =================================================
if page == "Route Planner":

    st.header("🚗 Eco-Friendly Route Planner")

    geo = Nominatim(user_agent="green_route_project")

    @st.cache_data
    def get_location(place):
        try:
            loc = geo.geocode(place, timeout=10)
            if loc:
                return (loc.latitude, loc.longitude)
        except:
            return None
        return None

    def haversine(a, b):
        R = 6371
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))

    @st.cache_resource
    def train_model():
        data = []
        for _ in range(500):
            d = random.uniform(2, 30)
            t = random.randint(1, 10)
            s = random.randint(0, 8)
            f = d*0.08 + t*0.03 + s*0.02
            data.append([d, t, s, f])
        df = pd.DataFrame(data, columns=["dist","traffic","stops","fuel"])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df[["dist","traffic","stops"]], df["fuel"])
        return model

    model = train_model()

    src_input = st.text_input("Source", "")
    dst_input = st.text_input("Destination", "")

    if st.button("Find Routes", type="primary"):

        src = get_location(src_input)
        dst = get_location(dst_input)

        if not src or not dst:
            st.error("Invalid locations")
            st.stop()

        base = round(haversine(src, dst), 2)

        rows = []
        for i in range(3):
            km = round(base + random.uniform(0,3), 2)
            traffic = random.randint(2,9)
            stops = random.randint(0,7)
            greenery = random.randint(45,95)

            X = pd.DataFrame([[km, traffic, stops]],
                             columns=["dist","traffic","stops"])

            fuel = round(model.predict(X)[0], 2)
            time = round((km/35)*60 + traffic*3, 1)

            eco = round((100 - traffic*5) + greenery*0.5 - fuel*8 - km, 2)

            rows.append({
                "Route": f"Route {i+1}",
                "Distance": km,
                "Time": time,
                "Fuel": fuel,
                "Eco Score": eco
            })

        df = pd.DataFrame(rows)

        best = df["Eco Score"].idxmax()

        st.success(f"🌿 Best Route: {df.loc[best,'Route']}")

        st.dataframe(df, width="stretch")
        st.bar_chart(df.set_index("Route")["Eco Score"])


# =================================================
# ================= MODULE 2 =======================
# =================================================
elif page == "Road Construction Planner":

    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    from scipy.interpolate import splprep, splev

    st.header("🏗 Eco-Friendly Road Construction Planner")

    geo = Nominatim(user_agent="road_planner")

    # ---------------------------
    # SESSION STATE
    # ---------------------------
    if "run_path" not in st.session_state:
        st.session_state.run_path = False

    if "grid" not in st.session_state:
        st.session_state.grid = None

    # ---------------------------
    # INPUT
    # ---------------------------
    start_place = st.text_input("Start Location", "")
    end_place = st.text_input("End Location", "")

    size = st.slider("Grid Size", 5, 20, 10)

    if st.button("Generate Road Path", type="primary"):
        st.session_state.run_path = True
        st.session_state.grid = None  # regenerate fresh grid

    # ---------------------------
    # GEOLOCATION
    # ---------------------------
    @st.cache_data
    def get_location(place):
        try:
            loc = geo.geocode(place, timeout=10)
            if loc:
                return (loc.latitude, loc.longitude)
        except:
            return None
        return None

    start_coords = get_location(start_place)
    end_coords = get_location(end_place)

    # ---------------------------
    # GRID
    # ---------------------------
    def generate_grid(n):
        return np.random.choice([1,3,6,100], size=(n,n),
                                p=[0.5,0.2,0.2,0.1])

    if st.session_state.grid is None:
        st.session_state.grid = generate_grid(size)

    grid = st.session_state.grid

    # ---------------------------
    # A* (Improved)
    # ---------------------------
    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def astar(grid, start, goal):
        pq = [(0, 0, start)]
        cost = {start: 0}
        path = {}

        while pq:
            _, _, current = heapq.heappop(pq)

            if current == goal:
                break

            for dx,dy in [
                (1,0),(-1,0),(0,1),(0,-1),
                (1,1),(1,-1),(-1,1),(-1,-1)
            ]:

                nx, ny = current[0]+dx, current[1]+dy

                if 0<=nx<len(grid) and 0<=ny<len(grid):

                    move_cost = 1.4 if dx != 0 and dy != 0 else 1
                    new_cost = cost[current] + grid[nx][ny] * move_cost

                    # TURN PENALTY
                    if current in path:
                        prev = path[current]
                        prev_dir = (current[0]-prev[0], current[1]-prev[1])
                        new_dir = (dx, dy)

                        if prev_dir != new_dir:
                            new_cost += 0.5

                    if (nx,ny) not in cost or new_cost < cost[(nx,ny)]:
                        cost[(nx,ny)] = new_cost
                        priority = new_cost + heuristic(goal,(nx,ny))
                        heapq.heappush(pq, (priority, heuristic(goal,(nx,ny)), (nx,ny)))
                        path[(nx,ny)] = current

        route = []
        cur = goal

        while cur in path:
            route.append(cur)
            cur = path[cur]

        route.append(start)
        route.reverse()

        return route, cost.get(goal, -1)

    # ---------------------------
    # PATH SMOOTHING
    # ---------------------------
    def smooth_path(path):
        if len(path) < 3:
            return path

        smooth = [path[0]]

        for i in range(1, len(path)-1):
            prev = smooth[-1]
            curr = path[i]
            nxt = path[i+1]

            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = nxt[0] - curr[0]
            dy2 = nxt[1] - curr[1]

            if (dx1, dy1) != (dx2, dy2):
                smooth.append(curr)

        smooth.append(path[-1])
        return smooth


    def create_smooth_curve(path):
        if len(path) < 3:
            return path

        x = [p[0] for p in path]
        y = [p[1] for p in path]

        tck, _ = splprep([x, y], s=2)
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)

        return list(zip(x_new, y_new))

    # ---------------------------
    # GRID → GEO
    # ---------------------------
    def grid_to_geo(path, start_lat, start_lon, end_lat, end_lon, size):
        coords = []
        lat_step = (end_lat - start_lat) / size
        lon_step = (end_lon - start_lon) / size

        for (x, y) in path:
            x = float(x)
            y = float(y)
            lat = start_lat + x * lat_step
            lon = start_lon + y * lon_step
            coords.append((lat, lon))

        return coords

    # ---------------------------
    # DISTANCE
    # ---------------------------
    def haversine(p1, p2):
        R = 6371
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # ---------------------------
    # RUN OUTPUT
    # ---------------------------
    if st.session_state.run_path:

        if not start_coords or not end_coords:
            st.error("Invalid location names")
            st.stop()

        start = (0,0)
        goal = (size-1, size-1)

        route, cost = astar(grid, start, goal)

        route = smooth_path(route)

        if len(route) > 3:
            route = create_smooth_curve(route)

        st.success(f"Optimal Construction Cost: {cost}")

        # GRID VISUAL
        st.subheader("📊 Terrain Simulation")

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap="terrain")

        x = [p[1] for p in route]
        y = [p[0] for p in route]

        ax.plot(x, y, color="red", linewidth=2)
        st.pyplot(fig)

        # MAP
        st.subheader("🗺 Proposed Highway Path")

        geo_path = grid_to_geo(route,
                               start_coords[0], start_coords[1],
                               end_coords[0], end_coords[1],
                               size)

        m = folium.Map(location=geo_path[0], zoom_start=13)

        folium.PolyLine(
            geo_path,
            color="red",
            weight=6,
            opacity=0.9,
            tooltip="Highway Optimized Route"
        ).add_to(m)

        folium.Marker(geo_path[0], tooltip="Start").add_to(m)
        folium.Marker(geo_path[-1], tooltip="End").add_to(m)

        m.fit_bounds(geo_path)

        st_folium(m, width=900, height=500)

        # DISTANCE
        total_dist = sum(
            haversine(geo_path[i], geo_path[i+1])
            for i in range(len(geo_path)-1)
        )

        st.info(f"📏 Estimated Road Length: {round(total_dist,2)} km")

        # ECO IMPACT
        eco_saved = round((100 - cost) * 0.05, 2)
        st.success(f"🌿 Environmental Benefit Score: {eco_saved}")

        # DOWNLOAD
        df_path = pd.DataFrame(geo_path, columns=["Latitude","Longitude"])
        st.download_button("📥 Download Path CSV",
                           df_path.to_csv(index=False),
                           "road_path.csv")
