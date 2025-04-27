import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import openrouteservice

# ===== LOAD DATA =====
df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
df_distance.index = df_distance.index.str.strip()
df_distance.columns = df_distance.columns.str.strip()
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

st.title("Villa Mart Optimal Route & Cost Optimizer (Robust VRP, OR-Tools)")

# --- INPUTS ---
st.header("Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    value = st.number_input(f"{store}", min_value=0, max_value=300, value=10)
    user_crate_demand[store] = value
    total_crates += value
st.write(f"**Total Crates Entered:** {total_crates}")

st.header("Truck, Rental & Cost Settings")
own_truck_a = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b = {"name": "Truck B", "capacity": 90, "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=200, value=121)
max_rented_trips = st.number_input("Max rented trips allowed", min_value=0, max_value=10, value=5)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)
petrol_price = st.number_input("Current petrol price (₹/litre)", min_value=0, max_value=1000, value=101)
mileage = 12.0  # fixed for all trucks

run_optim = st.button("Optimize Route")

# ======== OR-TOOLS VRP LOGIC =========
def create_data_model():
    demands = [0] + [user_crate_demand[s] for s in store_list[:-1]]  # depot first
    # Each trip = one vehicle in VRP
    vehicles = []
    for t in own_trucks:
        for i in range(t["max_trips"]):
            vehicles.append({"type": "own", "name": t["name"], "capacity": t["capacity"]})
    for i in range(max_rented_trips):
        vehicles.append({"type": "rented", "name": f"Rented-{i+1}", "capacity": rent_capacity})
    # Time windows and service times (optional, for now set to full window)
    time_windows = [(0, 240)] * len(store_list)
    service_times = [0] + [30 for _ in store_list[:-1]]
    # Build distance matrix in minutes (for routing cost, also for fuel)
    distance_matrix = []
    for from_store in store_list:
        row = []
        for to_store in store_list:
            km = df_distance.loc[from_store, to_store]
            time = int(km / avg_speed * 60)
            row.append(time)
        distance_matrix.append(row)
    data = {
        "distance_matrix": distance_matrix,
        "demands": demands,
        "vehicle_capacities": [v["capacity"] for v in vehicles],
        "num_vehicles": len(vehicles),
        "depot": 0,
        "vehicles": vehicles,
        "store_names": [depot] + store_list[:-1],
        "km_matrix": df_distance.values.tolist(),
        "service_times": service_times,
        "time_windows": time_windows,
        "avg_speed": avg_speed,
        "petrol_price": petrol_price,
        "mileage": mileage,
    }
    return data

def solve_vrp(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    # Distance callback
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Capacity
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        return data['demands'][from_node]
    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx, 0, data['vehicle_capacities'], True, 'Capacity')
    # Solve
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.time_limit.seconds = 30
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    solution = routing.SolveWithParameters(search_params)
    # Parse output
    trips = []
    if solution:
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            total_load = 0
            trip_km = 0
            stops = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    stops.append(data["store_names"][node])
                total_load += data['demands'][node]
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    trip_km += data["km_matrix"][manager.IndexToNode(prev_index)][manager.IndexToNode(index)]
            if stops:
                trips.append({
                    "vehicle_id": vehicle_id,
                    "vehicle": data["vehicles"][vehicle_id],
                    "route": stops,
                    "load": total_load,
                    "distance": trip_km,
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
        fuel_cost = (trip["distance"] / mileage) * petrol_price if not is_rented else 0
        total_cost = fuel_cost + rent_cost
        route_str = " -> ".join([f"{i+1}-{s} ({user_crate_demand.get(s, 0)})" for i, s in enumerate(trip["route"])])
        output.append({
            "truck": v["name"],
            "type": v["type"],
            "route": route_str,
            "load": trip["load"],
            "distance": round(trip["distance"], 2),
            "fuel_cost": round(fuel_cost, 2),
            "rent_cost": rent_cost,
            "total_cost": round(total_cost, 2),
        })
    df_output = pd.DataFrame(output)
    if not df_output.empty:
        st.header("Optimized Trip Plan (OR-Tools Robust)")
        st.dataframe(df_output)
        if len(output) > 0:
            trip_names = [f"{o['truck']}" for o in output]
            selected_trip = st.selectbox("Select truck to show its route:", trip_names)
            idx = trip_names.index(selected_trip)
            selected_stores = [depot] + trips[idx]["route"] + [depot]
            m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
            route_coords = [coords_dict[pt] for pt in selected_stores]
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
                        name=f"{selected_trip}",
                        style_function=lambda x: {'color': 'blue', 'weight': 6, 'opacity': 0.85}
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=route_coords[0][::-1],
                        radius=7, color="blue", fill=True, popup=selected_stores[0]
                    ).add_to(m)
            except Exception as ex:
                st.warning(f"Failed to plot real route for {selected_trip}: {ex}")
            for pt in selected_stores:
                folium.CircleMarker(
                    location=coords_dict[pt][::-1],
                    radius=10, color='blue', fill=True, popup=pt
                ).add_to(m)
            st.subheader(f"Route Visualization (Truck: {selected_trip})")
            st_folium(m, width=800, height=600)
    else:
        st.warning("No routes found. Please check your input.")
else:
    st.info("Enter demand and click Optimize to run.")
