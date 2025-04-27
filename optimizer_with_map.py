import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import itertools
import openrouteservice
import os

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

ORS_API_KEY = '5b3ce3597851110001cf62488858392856e24062ae6ba005c2e38325'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Bhubaneswar Route Optimizer (Truck/Trip Selector, Route/Crate Output)")

st.markdown("""
**Instructions:**  
1. Enter crate demand for each dark store.  
2. Set your truck and route constraints.  
3. Click "Optimize Route" to view optimal trip plan and routes.  
4. Use the dropdown to select a truck/trip and see only its route on the map!
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

run_optim = st.button("Optimize Route")

# --- If user clicks the button, compute and save to session_state ---
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

        remaining_demand = user_crate_demand.copy()
        trip_plan = []
        # --- Assign Own Trucks ---
        for truck in our_trucks:
            for t in range(truck["max_trips"]):
                load = 0
                stores_this_trip = []
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
                    st.session_state['last_results'] = None
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
                st.session_state['last_results'] = None
                st.stop()
            if rented_trip_count > max_rented_trips:
                st.error("Exceeded maximum allowed rented truck trips (possible infinite loop).")
                st.session_state['last_results'] = None
                st.stop()

        # --- Optimize Each Trip's Route (TSP or Greedy) ---
        results = []
        route_details_for_map = []
        for trip in trip_plan:
            # Prepare a dict to map store to crate for this trip
            store_to_crate = dict(trip["route_stores"])
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

            # Prepare route string for table: 1-Bomikhal (crates), 2-RTO (crates), ...
            route_ordered = []
            for stop_num, stop_name in enumerate(best_order, 1):
                crates = store_to_crate.get(stop_name, 0)
                route_ordered.append(f"{stop_num}-{stop_name} ({crates})")
            route_str = " -> ".join(route_ordered)

            # Save details for map
            route_details_for_map.append({
                "truck": trip["truck"],
                "trip_num": trip["trip_num"],
                "truck_type": trip["truck_type"],
                "route_stores": best_order,
                "store_to_crate": store_to_crate
            })

            results.append({
                "truck": trip["truck"],
                "trip_num": trip["trip_num"],
                "truck_type": trip["truck_type"],
                "route": route_str,
                "load": trip["load"],
                "distance": round(min_dist, 2),
                "total_time_min": total_time,
                "cost": cost
            })

        df_result = pd.DataFrame(results)
        if df_result.empty:
            st.warning("No routes generated. Please check your input values.")
            st.session_state['last_results'] = None
        else:
            st.session_state['last_results'] = {
                "df": df_result,
                "route_details_for_map": route_details_for_map
            }
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")
        st.session_state['last_results'] = None
        st.stop()

# --- Show output if present in session state ---
if 'last_results' in st.session_state and st.session_state['last_results'] is not None:
    df_last = st.session_state['last_results']['df']
    route_details_for_map = st.session_state['last_results']['route_details_for_map']
    if not df_last.empty:

        st.header("Optimized Trip Plan")
        st.dataframe(df_last)

        # Dropdown to select which trip/truck route to view
        truck_trip_options = [
            f"{d['truck']} - Trip {d['trip_num']}" for d in route_details_for_map
        ]
        selected_idx = 0
        if len(truck_trip_options) > 1:
            selected_idx = st.selectbox(
                "Select a Truck/Trip to view its route:",
                options=list(range(len(truck_trip_options))),
                format_func=lambda x: truck_trip_options[x],
            )
        selected_trip = route_details_for_map[selected_idx]

        # --- Route Visualization For Selected Truck/Trip Only ---
        m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
        color_cycle = [
            'blue', 'red', 'green', 'purple', 'orange', 'darkred', 
            'lightblue', 'gray', 'pink', 'black'
        ]
        color = color_cycle[selected_idx % len(color_cycle)]
        # Prepare route: depot -> stops in order -> depot
        route = [depot] + list(selected_trip["route_stores"]) + [depot]
        route_coords = [coords_dict[pt] for pt in route]
        # Query real route geometry from ORS
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
                    style_function=lambda x, color=color: {'color': color, 'weight': 5, 'opacity': 0.8}
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=route_coords[0][::-1],
                    radius=7, color=color, fill=True, popup=route[0]
                ).add_to(m)
        except Exception as ex:
            st.warning(f"Failed to plot real route for {selected_trip['truck']}: {ex}")

        # Plot markers for stops with crate info
        for stop_num, pt in enumerate(route):
            crate_val = ""
            if stop_num != 0 and stop_num != len(route)-1:
                crate_val = f"({selected_trip['store_to_crate'].get(pt, 0)} crates)"
            popup_str = f"{stop_num+1}: {pt} {crate_val}"
            tooltip_str = f"{stop_num+1}. {pt} {crate_val}"
            folium.CircleMarker(
                location=coords_dict[pt][::-1],
                radius=8, color=color, fill=True, popup=popup_str, tooltip=tooltip_str
            ).add_to(m)
            folium.map.Marker(
                coords_dict[pt][::-1],
                icon=folium.DivIcon(html=f"""<div style="font-size: 14pt; color : {color};"><b>{stop_num+1}</b></div>""")
            ).add_to(m)

        st.subheader(f"Route Visualization (Truck: {selected_trip['truck']}, Trip: {selected_trip['trip_num']})")
        st_folium(m, width=800, height=600)
    else:
        st.warning("No results to show. Please optimize again with new input.")
else:
    st.info("Enter input and click 'Optimize Route' to show results.")
