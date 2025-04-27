import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import itertools
import openrouteservice
from sklearn.cluster import KMeans
import numpy as np

# === LOAD DISTANCE MATRIX CSV ===
try:
    df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
    df_distance.index = df_distance.index.str.strip()
    df_distance.columns = df_distance.columns.str.strip()
except Exception as e:
    st.error(f"Error loading distance matrix: {e}")
    st.stop()

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

ORS_API_KEY = 'YOUR_ORS_KEY'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Bhubaneswar Route Optimizer (Petrol Cost for Own & Rental Trucks)")

st.markdown("""
**Instructions:**  
- Enter crate demand for each dark store  
- Set truck constraints and current petrol price  
- Click **Optimize Route**  
- See fuel cost for every own truck trip, rental cost for rented trucks.
""")

# ---- Inputs ----
st.header("Step 1: Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    value = st.number_input(f"{store}", min_value=0, max_value=300, value=10)
    user_crate_demand[store] = value
    total_crates += value
st.write(f"**Total Crates Entered:** {total_crates}")

st.header("Step 2: Truck Constraints & Cost Parameters")
our_trucks = [
    {"name": "Truck A", "capacity": 121, "max_trips": 2},
    {"name": "Truck B", "capacity": 90, "max_trips": 2},
]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=200, value=121)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)
petrol_price = st.number_input("Current petrol price (₹/litre)", min_value=0, max_value=1000, value=101)
mileage = 12.0  # fixed for all trucks

run_optim = st.button("Optimize Route")

# --- Efficient Clustering & Assignment ---
def efficient_trip_assignment(crate_demand, truck_list, rented_truck_capacity, max_rented_trips=15):
    stores_with_demand = [s for s, d in crate_demand.items() if d > 0]
    coords = np.array([coords_dict[s] for s in stores_with_demand])
    all_trips = []
    remaining_demand = crate_demand.copy()
    own_trips_remaining = []
    for truck in truck_list:
        for t in range(truck["max_trips"]):
            own_trips_remaining.append((truck["name"], truck["capacity"]))
    n_clusters = min(len(own_trips_remaining), len(stores_with_demand))
    if n_clusters == 0:
        return []
    if len(stores_with_demand) <= n_clusters:
        clusters = [[s] for s in stores_with_demand]
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(coords)
        cluster_ids = kmeans.labels_
        clusters = [[] for _ in range(n_clusters)]
        for idx, cid in enumerate(cluster_ids):
            clusters[cid].append(stores_with_demand[idx])
    for i, (truck_name, truck_capacity) in enumerate(own_trips_remaining):
        best_cluster_idx = -1
        best_total = 0
        for j, stores in enumerate(clusters):
            total_crates = sum(remaining_demand[s] for s in stores)
            if 0 < total_crates <= truck_capacity and total_crates > best_total:
                best_total = total_crates
                best_cluster_idx = j
        if best_cluster_idx == -1:
            continue
        chosen_stores = clusters[best_cluster_idx]
        trip = []
        for s in chosen_stores:
            trip.append((s, remaining_demand[s]))
            remaining_demand[s] = 0
        clusters[best_cluster_idx] = []
        all_trips.append({
            "truck": truck_name,
            "truck_type": "own",
            "capacity": truck_capacity,
            "route_stores": trip,
            "load": best_total
        })
    leftover_stores = [s for s in stores_with_demand if remaining_demand[s] > 0]
    trip_batches = []
    for s in leftover_stores:
        needed = remaining_demand[s]
        if needed > max([truck["capacity"] for truck in truck_list] + [rented_truck_capacity]):
            while needed > 0:
                take = min(needed, rented_truck_capacity)
                trip_batches.append([(s, take)])
                needed -= take
        else:
            trip_batches.append([(s, needed)])
    rented_trips = []
    for trip_batch in trip_batches:
        if len(rented_trips) >= max_rented_trips:
            break
        total_in_trip = sum(v for _, v in trip_batch)
        rented_trips.append({
            "truck": f"Rented-{len(rented_trips)+1}",
            "truck_type": "rented",
            "capacity": rented_truck_capacity,
            "route_stores": trip_batch,
            "load": total_in_trip
        })
    all_trips.extend(rented_trips)
    return all_trips

# --- Optimizer ---
if run_optim:
    try:
        if total_crates == 0:
            st.error("Total crate demand is zero. Please enter demand for at least one store.")
            st.session_state['last_results'] = None
            st.stop()
        if df_distance.isnull().values.any():
            st.error("Distance matrix contains missing values. Please check your CSV file.")
            st.session_state['last_results'] = None
            st.stop()

        trips = efficient_trip_assignment(user_crate_demand, our_trucks, rent_capacity)
        results = []
        trip_details = []
        for trip in trips:
            store_to_crate = dict(trip["route_stores"])
            route_stores = [store for store, _ in trip["route_stores"]]
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
            drive_time = (min_dist / avg_speed) * 60 if min_dist > 0 else 0
            wait_time = 30 * len(route_stores)
            fuel_cost = (min_dist / mileage) * petrol_price if trip["truck_type"] == "own" else 0
            rent_cost = 0 if trip["truck_type"] == "own" else (2300 if min_dist > 30 else 1500)
            total_cost = fuel_cost + rent_cost

            route_ordered = []
            for stop_num, stop_name in enumerate(best_order, 1):
                crates = store_to_crate.get(stop_name, 0)
                route_ordered.append(f"{stop_num}-{stop_name} ({crates})")
            route_str = " -> ".join(route_ordered)

            trip_details.append({
                "truck": trip["truck"],
                "trip_num": trips.index(trip)+1,
                "route_stores": list(best_order),
                "store_to_crate": store_to_crate
            })

            results.append({
                "truck": trip["truck"],
                "trip_type": trip["truck_type"],
                "route": route_str,
                "load": trip["load"],
                "distance": round(min_dist, 2),
                "fuel_cost": round(fuel_cost, 2),
                "rent_cost": rent_cost,
                "total_cost": round(total_cost, 2),
                "total_time_min": int(drive_time + wait_time),
            })

        df_result = pd.DataFrame(results)
        if df_result.empty:
            st.warning("No routes generated. Please check your input values.")
            st.session_state['last_results'] = None
        else:
            st.session_state['last_results'] = {
                "df": df_result,
                "trip_details": trip_details
            }
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
        st.session_state['last_results'] = None
        st.stop()

# --- Show output if present in session state ---
if 'last_results' in st.session_state and st.session_state['last_results'] is not None:
    df_last = st.session_state['last_results']['df']
    trip_details = st.session_state['last_results']['trip_details']
    if not df_last.empty:
        st.header("Optimized Trip Plan")
        st.dataframe(df_last)
        trip_names = [f"{trip_details[i]['truck']} Trip {trip_details[i]['trip_num']}" for i in range(len(trip_details))]
        selected_idx = 0
        if len(trip_names) > 1:
            selected_idx = st.selectbox(
                "Select truck/trip to show its route on the map:",
                options=list(range(len(trip_names))),
                format_func=lambda x: trip_names[x]
            )
        selected_trip = trip_details[selected_idx]
        m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
        color = 'blue'
        route = [depot] + selected_trip["route_stores"] + [depot]
        route_coords = [coords_dict[pt] for pt in route]
        try:
            if len(route_coords) > 1:
                response = ors_client.directions(
                    coordinates=route_coords,
                    profile='driving-hgv',
                    format='geojson'
                )
                geometry = response['features'][0]['geometry']
                folium.GeoJson(
                    geometry,
                    name=f"{selected_trip['truck']}",
                    style_function=lambda x, color=color: {'color': color, 'weight': 6, 'opacity': 0.85}
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=route_coords[0][::-1],
                    radius=7, color=color, fill=True, popup=route[0]
                ).add_to(m)
        except Exception as ex:
            st.warning(f"Failed to plot real route for {selected_trip['truck']}: {ex}")

        for pt in route:
            crate_val = ""
            if pt != depot:
                crate_val = f"({selected_trip['store_to_crate'].get(pt, 0)} crates)"
            popup_str = f"{pt} {crate_val}".strip()
            tooltip_str = popup_str
            folium.CircleMarker(
                location=coords_dict[pt][::-1],
                radius=10, color=color, fill=True, popup=popup_str, tooltip=tooltip_str
            ).add_to(m)
        st.subheader(f"Route Visualization (Truck: {selected_trip['truck']})")
        st_folium(m, width=800, height=600)
    else:
        st.warning("No results to show. Please optimize again with new input.")
else:
    st.info("Enter input and click 'Optimize Route' to show results.")
