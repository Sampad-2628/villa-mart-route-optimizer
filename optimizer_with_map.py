import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import openrouteservice

# ===== LOAD DATA =====
df_distance = pd.read_csv("bhubaneswar_truck_distance_matrix.csv", index_col=0)
df_distance.index = df_distance.index.str.strip()
df_distance.columns = df_distance.columns.str.strip()

# 1) Force your custom input order:
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
    "PatraPada",             # ← newly added
    "Villamart (Depot)"      # ← depot
]
depot = store_list[-1]

# 2) Coordinates dictionary (must match the same keys):
coords_dict = {
    "Sum Hospital":             [85.77249, 20.28318],
    "Aiims":                    [85.78570, 20.24993],
    "Bapuji Nagar":             [85.84149, 20.25109],
    "Baramunda":                [85.80296, 20.28139],
    "Bomikhal":                 [85.85578, 20.28113],
    "District Centre (CSPUR)":  [85.81827, 20.32664],
    "Kalinga":                  [85.81536, 20.29597],
    "Kanan Vihar - RTO":        [85.82865, 20.36067],
    "Patia":                    [85.82528, 20.35302],
    "Symphony Mall":            [85.88286, 20.32356],
    "PatraPada":                [85.76613, 20.23536],  # ← added
    "Villamart (Depot)":        [85.77015, 20.24394],
}

# ===== ORS CLIENT =====
ORS_API_KEY = "YOUR_ORS_API_KEY_HERE"
ors_client = openrouteservice.Client(key=ORS_API_KEY)

# ===== UI =====
st.title("Villa Mart Route Optimizer")

# Input: crate demand per store
st.header("Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    v = st.number_input(store, min_value=0, max_value=300, value=10)
    user_crate_demand[store] = v
    total_crates += v
st.write(f"**Total Crates Entered:** {total_crates}")

# Input: truck & cost settings
st.header("Truck, Rental & Cost Settings")
own_truck_a = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b = {"name": "Truck B", "capacity": 90,  "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=300, value=121)
max_rented_trips = st.number_input("Max rented trips allowed", min_value=0, max_value=10, value=5)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)
petrol_price = st.number_input("Current petrol price (₹/litre)", min_value=0, max_value=1000, value=101)
mileage = 12.0  # fixed

run_optim = st.button("Optimize Route")

# ===== DATA MODEL =====
def create_data_model():
    max_cap = max([t["capacity"] for t in own_trucks] + [rent_capacity])
    pod_expanded, pod_mapping, pod_demands, pod_coords = [], {}, [], []

    # split pods if demand > capacity
    for store in store_list[:-1]:
        demand = user_crate_demand[store]
        if demand > max_cap:
            full = demand // max_cap
            rem  = demand % max_cap
            for i in range(full):
                name = f"{store}__part{i+1}"
                pod_expanded.append(name)
                pod_mapping[name] = store
                pod_demands.append(max_cap)
                pod_coords.append(coords_dict[store])
            if rem:
                name = f"{store}__part{full+1}"
                pod_expanded.append(name)
                pod_mapping[name] = store
                pod_demands.append(rem)
                pod_coords.append(coords_dict[store])
        else:
            pod_expanded.append(store)
            pod_mapping[store] = store
            pod_demands.append(demand)
            pod_coords.append(coords_dict[store])

    all_points = [depot] + pod_expanded
    all_coords  = [coords_dict[depot]] + pod_coords

    # build expanded distance matrix
    dist_mat = []
    for f in all_points:
        row = []
        f_act = pod_mapping.get(f, depot)
        for t in all_points:
            t_act = pod_mapping.get(t, depot)
            row.append(df_distance.loc[f_act, t_act])
        dist_mat.append(row)

    # define vehicles
    vehicles, labels = [], []
    for t in own_trucks:
        for trip in range(1, t["max_trips"]+1):
            vehicles.append({"type":"own",    "name":t["name"], "capacity":t["capacity"], "trip":trip})
            labels.append(f"{t['name']} - Trip {trip}")
    for i in range(1, max_rented_trips+1):
        vehicles.append({"type":"rented", "name":f"Rented-{i}", "capacity":rent_capacity,  "trip":1})
        labels.append(f"Rented-{i}")

    return {
        "distance_matrix": dist_mat,
        "demands":         [0] + pod_demands,
        "vehicle_capacities":[v["capacity"] for v in vehicles],
        "num_vehicles":    len(vehicles),
        "depot":           0,
        "vehicles":        vehicles,
        "pod_names":       all_points,
        "pod_mapping":     pod_mapping,
        "coords":          all_coords,
        "truck_trip_labels":labels,
        "petrol_price":    petrol_price,
        "mileage":         mileage,
        "km_matrix":       dist_mat,
    }

# ===== SOLVER =====
def solve_vrp(data):
    mgr   = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    model = pywrapcp.RoutingModel(mgr)

    # discourage rented vehicles by high fixed cost
    for vid, v in enumerate(data["vehicles"]):
        if v["type"] == "rented":
            model.SetFixedCostOfVehicle(100000, vid)

    def dist_cb(i, j):
        return int(data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)] * 100)
    dc = model.RegisterTransitCallback(dist_cb)
    model.SetArcCostEvaluatorOfAllVehicles(dc)

    def dem_cb(i):
        return data["demands"][mgr.IndexToNode(i)]
    dcb = model.RegisterUnaryTransitCallback(dem_cb)
    model.AddDimensionWithVehicleCapacity(dcb, 0, data["vehicle_capacities"], True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 30
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = model.SolveWithParameters(params)
    trips = []
    if sol:
        for vid in range(data["num_vehicles"]):
            idx = model.Start(vid)
            route, load, dist = [], 0, 0
            while not model.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node != 0:
                    route.append(data["pod_names"][node])
                load += data["demands"][node]
                prev = idx
                idx = sol.Value(model.NextVar(idx))
                if not model.IsEnd(idx):
                    dist += data["km_matrix"][mgr.IndexToNode(prev)][mgr.IndexToNode(idx)]
            if route:
                trips.append({
                    "vehicle_id": vid,
                    "vehicle":    data["vehicles"][vid],
                    "route":      route,
                    "load":       load,
                    "distance":   dist,
                    "label":      data["truck_trip_labels"][vid]
                })
    return trips

# ===== RUN & OUTPUT =====
if run_optim:
    data  = create_data_model()
    trips = solve_vrp(data)
    output = []
    for trip in trips:
        v      = trip["vehicle"]
        rented = (v["type"] == "rented")
        rent_c = 2300 if rented and trip["distance"]>30 else (1500 if rented else 0)
        fuel_c = (trip["distance"]/data["mileage"])*data["petrol_price"] if not rented else 0
        total  = rent_c + fuel_c

        route_str = " -> ".join(
            f"{i+1}-{data['pod_mapping'].get(s,s)} ({data['demands'][data['pod_names'].index(s)]})"
            for i, s in enumerate(trip["route"])
        )
        output.append({
            "truck":      v["name"],
            "trip":       v["trip"],
            "type":       v["type"],
            "label":      trip["label"],
            "route":      route_str,
            "load":       trip["load"],
            "distance":   round(trip["distance"], 2),
            "fuel_cost":  round(fuel_c, 2),
            "rent_cost":  rent_c,
            "total_cost": round(total, 2),
        })

    df_out = pd.DataFrame(output)
    st.session_state['vrp_df_output'] = df_out
    st.session_state['vrp_trips']     = trips
    st.session_state['vrp_data']      = data

if 'vrp_df_output' in st.session_state and not st.session_state['vrp_df_output'].empty:
    df_out = st.session_state['vrp_df_output']
    trips  = st.session_state['vrp_trips']
    data   = st.session_state['vrp_data']

    st.header("Optimized Trip Plan")
    st.dataframe(df_out)

    sel = st.selectbox("Select trip to visualize:", df_out['label'].tolist())
    idx = df_out.index[df_out['label']==sel][0]
    trip = trips[idx]
    stops = [depot] + trip["route"] + [depot]

    m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
    coords = [coords_dict[data["pod_mapping"].get(s,s)] for s in stops]

    try:
        if len(coords)>1:
            resp = ors_client.directions(coordinates=coords, profile='driving-hgv', format='geojson')
            geom = resp['features'][0]['geometry']
            folium.GeoJson(geom, style_function=lambda x:{'color':'blue','weight':6,'opacity':0.8}).add_to(m)
        else:
            folium.CircleMarker(location=coords[0][::-1], radius=7, color='blue', fill=True).add_to(m)
    except Exception as e:
        st.warning(f"Route plot failed: {e}")

    for s, coord in zip(stops, coords):
        lbl = f"{data['pod_mapping'].get(s,s)}: {data['demands'][data['pod_names'].index(s)]} crates"
        folium.CircleMarker(location=coord[::-1], radius=8, color='blue', fill=True, tooltip=lbl).add_to(m)

    st.subheader(f"Route Map: {sel}")
    st_folium(m, width=800, height=600)
else:
    st.info("Enter demands and click Optimize to run.")
