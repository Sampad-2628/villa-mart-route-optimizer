
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import openrouteservice

# ===== LOAD DISTANCE MATRIX =====
df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
df_distance.index = df_distance.index.str.strip()
df_distance.columns = df_distance.columns.str.strip()

# ===== FORCE STORE ORDER INCLUDING NEW DARKSTORE =====
store_list = [
    "Sum Hospital",
    "Aiims",
    "Bapuji Nagar",
    "Baramunda",
    "Bomikhal",
    "District Centre (CSPUR)",
    "Kalinga",
    "Kanan Vihar - RTO",
    "Patia",
    "Symphony Mall",
    "Patrapada",
    "Villamart (Depot)"
]
depot = store_list[-1]

# ===== COORDINATES =====
coords_dict = {
    "Sum Hospital": [85.77249, 20.28318],
    "Aiims": [85.78570, 20.24993],
    "Bapuji Nagar": [85.84149, 20.25109],
    "Baramunda": [85.80296, 20.28139],
    "Bomikhal": [85.85578, 20.28113],
    "District Centre (CSPUR)": [85.81827, 20.32664],
    "Kalinga": [85.81536, 20.29597],
    "Kanan Vihar - RTO": [85.82865, 20.36067],
    "Patia": [85.82528, 20.35302],
    "Symphony Mall": [85.88286, 20.32356],
    "Patrapada": [85.76613, 20.23536],
    "Villamart (Depot)": [85.77015, 20.24394],
}

ORS_API_KEY = '5b3ce3597851110001cf62488858392856e24062ae6ba005c2e38325'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Villa Mart Route Optimizer (Smart Pod-Splitting, Cost & Utilization)")

# ===== CRATE DEMAND INPUT =====
st.header("Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    try:
        value = int(st.number_input(f"{store}", min_value=0, max_value=300, value=10))
    except:
        value = 0
    user_crate_demand[store] = value
    total_crates += value
st.write(f"**Total Crates Entered:** {total_crates}")

# ===== TRUCK SETTINGS =====
st.header("Truck, Rental & Cost Settings")
own_truck_a = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b = {"name": "Truck B", "capacity": 90, "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=300, value=121)
max_rented_trips = st.number_input("Max rented trips allowed", min_value=0, max_value=10, value=5)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)
petrol_price = st.number_input("Current petrol price (₹/litre)", min_value=0, max_value=1000, value=101)
mileage = 12.0

run_optim = st.button("Optimize Route")

# ===== DATA MODEL CREATION =====
def create_data_model():
    max_truck_capacity = max([t["capacity"] for t in own_trucks] + [rent_capacity])
    pod_expanded, pod_mapping, pod_demands, pod_coords = [], {}, [], []

    for store in store_list[:-1]:
        demand = user_crate_demand.get(store, 0)
        if demand is None:
            demand = 0
        if demand > max_truck_capacity:
            n_full = demand // max_truck_capacity
            last = demand % max_truck_capacity
            for i in range(n_full):
                new_name = f"{store}__part{i+1}"
                pod_expanded.append(new_name)
                pod_mapping[new_name] = store
                pod_demands.append(max_truck_capacity)
                pod_coords.append(coords_dict[store])
            if last > 0:
                new_name = f"{store}__part{n_full+1}"
                pod_expanded.append(new_name)
                pod_mapping[new_name] = store
                pod_demands.append(last)
                pod_coords.append(coords_dict[store])
        else:
            pod_expanded.append(store)
            pod_mapping[store] = store
            pod_demands.append(demand)
            pod_coords.append(coords_dict[store])

    all_points = [depot] + pod_expanded
    all_coords = [coords_dict[depot]] + pod_coords
    expanded_dist = []
    for from_store in all_points:
        from_actual = pod_mapping.get(from_store, from_store)
        row = []
        for to_store in all_points:
            to_actual = pod_mapping.get(to_store, to_store)
            row.append(df_distance.loc[from_actual, to_actual])
        expanded_dist.append(row)

    vehicles = []
    truck_trip_labels = []
    for t in own_trucks:
        for i in range(t["max_trips"]):
            vehicles.append({"type": "own", "name": t["name"], "capacity": t["capacity"], "trip": i+1})
            truck_trip_labels.append(f"{t['name']} - Trip {i+1}")
    for i in range(max_rented_trips):
        vehicles.append({"type": "rented", "name": f"Rented-{i+1}", "capacity": rent_capacity, "trip": 1})
        truck_trip_labels.append(f"Rented-{i+1}")

    return {
        "distance_matrix": expanded_dist,
        "demands": [0] + pod_demands,
        "vehicle_capacities": [v["capacity"] for v in vehicles],
        "num_vehicles": len(vehicles),
        "depot": 0,
        "vehicles": vehicles,
        "pod_names": all_points,
        "display_names": [depot] + pod_expanded,
        "pod_mapping": pod_mapping,
        "coords": all_coords,
        "truck_trip_labels": truck_trip_labels,
        "petrol_price": petrol_price,
        "mileage": mileage,
        "km_matrix": expanded_dist,
    }

def solve_vrp(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    for vehicle_id, v in enumerate(data['vehicles']):
        if v['type'] == 'rented':
            routing.SetFixedCostOfVehicle(100000, vehicle_id)

    def distance_callback(from_idx, to_idx):
        return int(data['distance_matrix'][manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] * 100)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_idx):
        return data['demands'][manager.IndexToNode(from_idx)]
    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx, 0, data['vehicle_capacities'], True, 'Capacity')

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.seconds = 30
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    solution = routing.SolveWithParameters(search_params)
    trips = []

    if solution:
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            total_load = 0
            trip_km = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(data["pod_names"][node])
                total_load += data['demands'][node]
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    trip_km += data["km_matrix"][manager.IndexToNode(prev_index)][manager.IndexToNode(index)]
            if route:
                trips.append({
                    "vehicle_id": vehicle_id,
                    "vehicle": data["vehicles"][vehicle_id],
                    "route": route,
                    "load": total_load,
                    "distance": trip_km / 1.0,
                    "label": data["truck_trip_labels"][vehicle_id]
                })
    return trips

if run_optim:
    data = create_data_model()
    trips = solve_vrp(data)
    output = []
    for trip in trips:
        v = trip["vehicle"]
        is_rented = v["type"] == "rented"
        rent_cost = 2300 if is_rented and trip["distance"] > 30 else 1500 if is_rented else 0
        fuel_cost = (trip["distance"] / data["mileage"]) * data["petrol_price"] if not is_rented else 0
        total_cost = fuel_cost + rent_cost
        output_route = []
        for i, s in enumerate(trip["route"]):
            real_store = data["pod_mapping"].get(s, s)
            crates = data['demands'][data["pod_names"].index(s)]
            label = f"{i+1}-{real_store} (split, {crates})" if s != real_store else f"{i+1}-{real_store} ({crates})"
            output_route.append(label)
        output.append({
            "truck": v["name"],
            "trip": v["trip"],
            "label": trip["label"],
            "type": v["type"],
            "route": " -> ".join(output_route),
            "load": trip["load"],
            "distance": round(trip["distance"], 2),
            "fuel_cost": round(fuel_cost, 2),
            "rent_cost": rent_cost,
            "total_cost": round(total_cost, 2),
        })

    df_output = pd.DataFrame(output)
    st.session_state['vrp_df_output'] = df_output
    st.session_state['vrp_trips'] = trips
    st.session_state['vrp_output_data'] = output
    st.session_state['vrp_trip_labels'] = [o['label'] for o in output]
    st.session_state['vrp_data'] = data

if 'vrp_df_output' in st.session_state and not st.session_state['vrp_df_output'].empty:
    df_output = st.session_state['vrp_df_output']
    trips = st.session_state['vrp_trips']
    output = st.session_state['vrp_output_data']
    trip_labels = st.session_state['vrp_trip_labels']
    data = st.session_state['vrp_data']

    st.header("Optimized Trip Plan")
    st.dataframe(df_output)

    selected_trip_label = st.selectbox("Select truck/trip to show its route:", trip_labels)
    idx = trip_labels.index(selected_trip_label)
    selected_trip = trips[idx]
    selected_stores = [depot] + selected_trip["route"] + [depot]
    m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
    route_coords = [coords_dict[data["pod_mapping"].get(pt, pt)] for pt in selected_stores]

    hover_labels = []
    for pt in selected_stores:
        if pt == depot:
            hover_labels.append(f"{depot} (Depot, 0)")
        else:
            idx_pt = data["pod_names"].index(pt)
            real = data["pod_mapping"].get(pt, pt)
            crates = data["demands"][idx_pt]
            hover_labels.append(f"{real}: {crates} crates")

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
                name=selected_trip_label,
                style_function=lambda x: {'color': 'blue', 'weight': 6, 'opacity': 0.85}
            ).add_to(m)
    except Exception as ex:
        st.warning(f"Failed to plot real route: {ex}")

    for pt, label, coord in zip(selected_stores, hover_labels, route_coords):
        folium.CircleMarker(
            location=coord[::-1],
            radius=10,
            color='blue',
            fill=True,
            popup=label,
            tooltip=label
        ).add_to(m)

    st.subheader(f"Route Visualization ({selected_trip_label})")
    st_folium(m, width=800, height=600)
else:
    st.info("Enter demand and click Optimize to run.")

