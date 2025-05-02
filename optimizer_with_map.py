import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import openrouteservice

# ===== LOAD DISTANCE MATRIX =====
df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
df_distance.index   = df_distance.index.str.strip()
df_distance.columns = df_distance.columns.str.strip()

# ===== STORE ORDER & DEPOT =====
# (make sure this matches exactly what you prompt below)
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
    "PatraPada",           # ← newly added
    "Villamart (Depot)"    # ← depot
]
depot = store_list[-1]

# ===== COORDINATES DICT =====
coords_dict = {
    "Sum Hospital":            [85.77249, 20.28318],
    "Aiims":                   [85.78570, 20.24993],
    "Bapuji Nagar":            [85.84149, 20.25109],
    "Baramunda":               [85.80296, 20.28139],
    "Bomikhal":                [85.85578, 20.28113],
    "District Centre (CSPUR)": [85.81827, 20.32664],
    "Kalinga":                 [85.81536, 20.29597],
    "Kanan Vihar - RTO":       [85.82865, 20.36067],
    "Patia":                   [85.82528, 20.35302],
    "Symphony Mall":           [85.88286, 20.32356],
    "PatraPada":               [85.76613, 20.23536],  # ← added
    "Villamart (Depot)":       [85.77015, 20.24394],
}

# ===== ORS CLIENT =====
ORS_API_KEY = "5b3ce3597851110001cf62488858392856e24062ae6ba005c2e38325"
ors_client   = openrouteservice.Client(key=ORS_API_KEY)

# ===== UI HEADER =====
st.title("Villa Mart Route Optimizer (Smart Pod-Splitting, Cost & Utilization)")

# ===== USER INPUTS =====
st.header("Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    v = st.number_input(store, min_value=0, max_value=300, value=10)
    user_crate_demand[store] = v
    total_crates += v
st.write(f"**Total Crates Entered:** {total_crates}")

st.header("Truck, Rental & Cost Settings")
own_truck_a      = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b      = {"name": "Truck B", "capacity": 90,  "max_trips": 2}
own_trucks       = [own_truck_a, own_truck_b]
rent_capacity    = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=300, value=121)
max_rented_trips = st.number_input("Max rented trips allowed",       min_value=0,  max_value=10, value=5)
avg_speed        = st.number_input("Average truck speed (km/hr)",     min_value=30, max_value=100, value=60)
petrol_price     = st.number_input("Current petrol price (₹/litre)",  min_value=0,  max_value=1000,value=101)
mileage          = 12.0  # fixed

run_optim = st.button("Optimize Route")

# ===== DATA MODEL WITH DEBUG =====
def create_data_model():
    # --- DEBUGGING LINES ---
    st.write("DEBUG ➜ store_list[:-1] =", store_list[:-1])
    st.write("DEBUG ➜ user_crate_demand.keys() =", list(user_crate_demand.keys()))
    # -------------------------

    max_cap      = max([t["capacity"] for t in own_trucks] + [rent_capacity])
    pod_exp, pod_map, pod_dem, pod_coords = [], {}, [], []

    for store in store_list[:-1]:
        demand = user_crate_demand[store]   # <-- this is where KeyError occurred
        if demand > max_cap:
            full = demand // max_cap
            rem  = demand % max_cap
            for i in range(full):
                name = f"{store}__part{i+1}"
                pod_exp.append(name)
                pod_map[name] = store
                pod_dem.append(max_cap)
                pod_coords.append(coords_dict[store])
            if rem:
                name = f"{store}__part{full+1}"
                pod_exp.append(name)
                pod_map[name] = store
                pod_dem.append(rem)
                pod_coords.append(coords_dict[store])
        else:
            pod_exp.append(store)
            pod_map[store] = store
            pod_dem.append(demand)
            pod_coords.append(coords_dict[store])

    all_pts    = [depot] + pod_exp
    all_coords = [coords_dict[depot]] + pod_coords

    # build expanded distance matrix
    dist_mat = []
    for f in all_pts:
        f_act = pod_map.get(f, depot)
        row = []
        for t in all_pts:
            t_act = pod_map.get(t, depot)
            row.append(df_distance.loc[f_act, t_act])
        dist_mat.append(row)

    # vehicle definitions
    vehicles, labels = [], []
    for t in own_trucks:
        for trip in range(1, t["max_trips"]+1):
            vehicles.append({"type":"own", "name":t["name"],   "capacity":t["capacity"], "trip":trip})
            labels.append(f"{t['name']} - Trip {trip}")
    for i in range(1, max_rented_trips+1):
        vehicles.append({"type":"rented", "name":f"Rented-{i}", "capacity":rent_capacity, "trip":1})
        labels.append(f"Rented-{i}")

    return {
        "distance_matrix": dist_mat,
        "demands":         [0] + pod_dem,
        "vehicle_capacities":[v["capacity"] for v in vehicles],
        "num_vehicles":    len(vehicles),
        "depot":           0,
        "vehicles":        vehicles,
        "pod_names":       all_pts,
        "pod_mapping":     pod_map,
        "coords":          all_coords,
        "truck_trip_labels":labels,
        "petrol_price":    petrol_price,
        "mileage":         mileage,
        "km_matrix":       dist_mat,
    }

# ===== VRP SOLVER & MAP CODE REMAINS UNCHANGED =====
# (Paste your existing solve_vrp(...) and the Streamlit output/map logic here)
