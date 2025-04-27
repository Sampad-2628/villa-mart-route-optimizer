import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import itertools

# === LOAD DISTANCE MATRIX CSV ===
try:
    # Strip all whitespace from names (prevents lookup bugs)
    df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
    df_distance.index = df_distance.index.str.strip()
    df_distance.columns = df_distance.columns.str.strip()
except Exception as e:
    st.error(f"Error loading distance matrix: {e}")
    st.stop()

# Store names and depot
store_list = df_distance.index.tolist()
depot = store_list[-1]

coords_dict = {
    "Kanan Vihar - RTO": [85.82865, 20.36067],
    "Sum Hospital": [85.77249, 20.28318],
    "Symphony Mall": [85.88286, 20.32356],
    "Aiims": [85.78570, 20.24993],
    "Bapuji Nagar": [85.84149, 20.25109],
    "Baramunda": [85.80296, 20.28139],
    "District Centre (CSPUR)": [85.81827, 20.32664],
    "Patia": [85.82528, 20.35302],
    "Kalinga": [85.81536, 20.29597],
    "Bomikhal": [85.85578, 20.28113],
    "Villamart (Depot)": [85.77015, 20.24394],
}

# --- Streamlit Interface ---
st.title("Bhubaneswar Route Optimizer (Robust & Safe Version)")
st.markdown("""
**Instructions:**  
1. Enter crate demand for each dark store.  
2. Set your truck and route constraints.  
3. Click "Optimize Route" to view optimal trip plan and routes.
""")

# -- UI Inputs --
st.header("Step 1: Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    value = st.number_input(f"{store}", min_value=0, max_value=300, value=10)
    user_crate_demand[store] = value
    total_crates += value
st.write(f"**Total Crates Entered:** {total_crates}")

st.header("Step 2: Truck Constraints")
our_trucks = [
    {"name": "Truck A", "capacity": 121, "max_trips": 2},
    {"name": "Truck B", "capacity": 90, "max_trips": 2},
]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=200, value=121)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)

if st.button("Optimize Route"):
    try:
        if total_crates == 0:
            st.error("Total crate demand is zero. Please enter demand for at least one store.")
            st.stop()

        if df_distance.isnull().values.any():
            st.error("Distance matrix contains missing values. Please check your CSV file.")
            st.stop()

        remaining_demand = user_crate_demand.copy()
        trip_plan = []
        # --- Assign Own Trucks ---
        for truck in our_trucks:
            for t in range(truck["max_trips"]):
                load = 0
                stores_this_trip = []
                # Assign stores with highest remaining demand first
                for store, crates in sorted(remaining_demand.items(), key=lambda x: -x[1]):
                    if crates == 0:
                        continue
                    can_assign = min(crates, truck["capacity"] - load)
                    if can_assign > 0:
                        stores_this_trip.append((store, can_assign))
                        load += can_assign
                        remaining_demand[store] -= can_assign
                    if load == truck["capacity"]:
                        break
                if load > 0:
                    trip_plan.append({
                        "truck": truck["name"],
                        "trip_num": t + 1,
                        "truck_type": "own",
                        "capacity": truck["capacity"],
                        "route_stores": stores_this_trip,
                        "load": load
                    })
        # --- Assign Rented Trucks ---
        max_rented_trips = 100
        rented_trip_count = 0
        while sum(remaining_demand.values()) > 0:
            trip_load = 0
            stores_in_trip = []
            for store, crates in sorted(remaining_demand.items(), key=lambda x: -x[1]):
                if crates == 0:
                    continue
                if crates > rent_capacity:
                    st.error(
                        f"Store '{store}' crate demand ({crates}) exceeds rented truck capacity ({rent_capacity})."
                        " Split the demand or increase rented truck capacity."
                    )
                    st.stop()
                can_assign = min(crates, rent_capacity - trip_load)
                if can_assign > 0:
                    stores_in_trip.append((store, can_assign))
                    trip_load += can_assign
                    remaining_demand[store] -= can_assign
                if trip_load == rent_capacity:
                    break
            rented_trip_count += 1
            if trip_load > 0:
                trip_plan.append({
                    "truck": f"Rented-{rented_trip_count}",
                    "trip_num": rented_trip_count,
                    "truck_type": "rented",
                    "capacity": rent_capacity,
                    "route_stores": stores_in_trip,
                    "load": trip_load
                })
            else:
                st.error("Infinite loop detected: cannot allocate crates to rented trucks. "
                         "Check your demand/capacity settings.")
                st.stop()
            if rented_trip_count > max_rented_trips:
                st.error("Exceeded maximum allowed rented truck trips (possible infinite loop).")
                st.stop()

        # --- Optimize Each Trip's Route (TSP or Greedy) ---
        results = []
        for trip in trip_plan:
            route_stores = [store for store, _ in trip["route_stores"]]
            route_stores = list(set(route_stores))  # Deduplicate
            if not route_stores:
                continue
            best_order = route_stores
            min_dist = float('inf')
            if len(route_stores) <= 7:
                for perm in itertools.permutations(route_stores):
                    dist = df_distance.loc[depot, perm[0]]
                    for i in range(len(perm)-1):
                        dist += df_distance.loc[perm[i], perm[i+1]]
                    dist += df_distance.loc[perm[-1], depot]
                    if dist < min_dist:
                        min_dist = dist
                        best_order = perm
            else:
                # Greedy nearest neighbor for larger trips
                best_order = []
                unvisited = set(route_stores)
                current = depot
                total_dist = 0
                while unvisited:
                    next_store = min(unvisited, key=lambda x: df_distance.loc[current, x])
                    best_order.append(next_store)
                    total_dist += df_distance.loc[current, next_store]
                    current = next_store
                    unvisited.remove(next_store)
                total_dist += df_distance.loc[current, depot]
                min_dist = total_dist

            drive_time = (min_dist / avg_speed) * 60 if min_dist > 0 else 0  # in minutes
            wait_time = 30 * len(route_stores)  # 30 min per store
            total_time = int(drive_time + wait_time)
            cost = 0 if trip["truck_type"] == "own" else (2300 if min_dist > 30 else 1500)
            results.append({
                "truck": trip["truck"],
                "trip_num": trip["trip_num"],
                "truck_type": trip["truck_type"],
                "stores": best_order,
                "load": trip["load"],
                "distance": round(min_dist, 2),
                "total_time_min": total_time,
                "cost": cost
            })
        df_result = pd.DataFrame(results)
        if df_result.empty:
            st.warning("No routes generated. Please check your input values.")
        else:
            st.header("Optimized Trip Plan")
            st.dataframe(df_result)

            # --- Route Visualization ---
            m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
            color_cycle = [
                'blue', 'red', 'green', 'purple', 'orange', 'darkred', 
                'lightblue', 'gray', 'pink', 'black'
            ]
            for idx, row in df_result.iterrows():
                color = color_cycle[idx % len(color_cycle)]
                route = [depot] + list(row["stores"]) + [depot]
                route_coords = [coords_dict[pt] for pt in route]
                folium.PolyLine(
                    locations=[c[::-1] for c in route_coords],
                    color=color, weight=5, opacity=0.7, tooltip=f"{row['truck']}"
                ).add_to(m)
                for pt in route:
                    folium.CircleMarker(
                        location=coords_dict[pt][::-1],
                        radius=6, color=color, fill=True, popup=pt
                    ).add_to(m)
            st.subheader("Route Visualization")
            st_folium(m, width=800, height=600)

    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
        st.stop()

# ---- LOGIC (in plain English, outside the code block) ----

# LOGIC EXPLANATION:
# 1. User enters crate demand and truck constraints.
# 2. Code assigns crates to own trucks first (using capacity and trips), then rented trucks (with protection against infinite loop).
# 3. Each trip's stops are optimized using brute-force TSP if <=7 stops, or greedy nearest neighbor otherwise.
# 4. The app outputs a DataFrame of all trips (truck, stops, distance, time, cost) and plots the routes on a map.
# 5. Robust error handling is provided for all possible bad inputs or data issues.

