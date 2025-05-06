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
store_list = df_distance.index.tolist()
depot = store_list[-1]

coords_dict = {
    "Kanan Vihar - RTO":      [85.82865, 20.36067],
    "Sum Hospital":           [85.77249, 20.28318],
    "Symphony Mall":          [85.88286, 20.32356],
    "Aiims":                  [85.78570, 20.24993],
    "Bapuji Nagar":           [85.84149, 20.25109],
    "Baramunda":              [85.80296, 20.28139],
    "District Centre (CSPUR)": [85.81827, 20.32664],
    "Patia":                  [85.82528, 20.35302],
    "Kalinga":                [85.81536, 20.29597],
    "Bomikhal":               [85.85578, 20.28113],
    "PatraPada":              [85.76613, 20.23536],
    "Villamart (Depot)":      [85.77015, 20.24394],
}

ORS_API_KEY = 'YOUR_ORS_API_KEY'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Villa Mart Route Optimizer")

# --- INPUTS ---
st.header("Enter Crate Demand for Each Store")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    val = st.number_input(store, min_value=0, max_value=300, value=10)
    user_crate_demand[store] = val
    total_crates += val
st.write(f"**Total Crates**: {total_crates}")

st.header("Truck Settings")
own_truck_a = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b = {"name": "Truck B", "capacity": 90,  "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity = st.number_input("Rented truck capacity", 70, 300, 121)
max_rented_trips = st.number_input("Max rented trips", 0, 10, 5)
petrol_price = st.number_input("Petrol price (₹/L)", 0, 1000, 101)
mileage = 12.0

run_optim = st.button("Optimize Route")

def create_data_model():
    max_cap = max([t["capacity"] for t in own_trucks] + [rent_capacity])
    pod_expanded, pod_mapping, pod_demands, pod_coords = [], {}, [], []
    for store in store_list[:-1]:
        d = user_crate_demand[store]
        if d > max_cap:
            full, rem = divmod(d, max_cap)
            for i in range(full):
                n = f"{store}__part{i+1}"
                pod_expanded.append(n); pod_mapping[n]=store
                pod_demands.append(max_cap); pod_coords.append(coords_dict[store])
            if rem:
                n = f"{store}__part{full+1}"
                pod_expanded.append(n); pod_mapping[n]=store
                pod_demands.append(rem); pod_coords.append(coords_dict[store])
        else:
            pod_expanded.append(store); pod_mapping[store]=store
            pod_demands.append(d); pod_coords.append(coords_dict[store])

    all_points = [depot] + pod_expanded
    expanded_dist = []
    for src in all_points:
        src_act = pod_mapping.get(src, src)
        row = [df_distance.loc[src_act, pod_mapping.get(dst, dst)] for dst in all_points]
        expanded_dist.append(row)

    # build vehicles
    vehicles, labels = [], []
    for t in own_trucks:
        for trip in range(t["max_trips"]):
            vehicles.append({"type":"own","name":t["name"],"capacity":t["capacity"],"trip":trip+1})
            labels.append(f"{t['name']} - Trip {trip+1}")
    for i in range(max_rented_trips):
        vehicles.append({"type":"rented","name":f"Rented-{i+1}","capacity":rent_capacity,"trip":1})
        labels.append(f"Rented-{i+1}")

    return {
        "distance_matrix": expanded_dist,
        "demands": [0]+pod_demands,
        "vehicle_capacities": [v["capacity"] for v in vehicles],
        "num_vehicles": len(vehicles),
        "depot": 0,
        "vehicles": vehicles,
        "pod_names": all_points,
        "pod_mapping": pod_mapping,
        "truck_trip_labels": labels,
        "petrol_price": petrol_price,
        "mileage": mileage,
        "km_matrix": expanded_dist,
    }

def solve_vrp(data):
    mgr = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(mgr)

    # fixed cost for rented
    for vid,v in enumerate(data["vehicles"]):
        if v["type"]=="rented":
            routing.SetFixedCostOfVehicle(100000, vid)

    def dist_cb(i,j):
        return int(data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)] * 100)
    t_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(t_cb)

    def dem_cb(i):
        return data["demands"][mgr.IndexToNode(i)]
    d_cb = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(d_cb, 0, data["vehicle_capacities"], True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 30
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = routing.SolveWithParameters(params)
    trips = []
    if sol:
        for vid in range(data["num_vehicles"]):
            idx, route, load, dist_km = routing.Start(vid), [], 0, 0
            while not routing.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node!=0:
                    route.append(data["pod_names"][node])
                load += data["demands"][node]
                prev, idx = idx, sol.Value(routing.NextVar(idx))
                if not routing.IsEnd(idx):
                    dist_km += data["km_matrix"][mgr.IndexToNode(prev)][mgr.IndexToNode(idx)]
            if route:
                trips.append({
                    "vehicle_id": vid,
                    "vehicle": data["vehicles"][vid],
                    "route": route,
                    "load": load,
                    "distance": dist_km,
                    "label": data["truck_trip_labels"][vid],  # ← ensure every trip has a 'label'
                })
    return trips

if run_optim:
    data  = create_data_model()
    trips = solve_vrp(data)
    output = []

    for trip in trips:
        v = trip["vehicle"]
        is_r = (v["type"]=="rented")
        rent_cost = 2300 if is_r and trip["distance"]>30 else (1500 if is_r else 0)
        fuel_cost = (trip["distance"]/data["mileage"])*data["petrol_price"] if not is_r else 0
        total = round(rent_cost + fuel_cost, 2)

        # human‐friendly route string
        parts = []
        for i,stop in enumerate(trip["route"]):
            real = data["pod_mapping"].get(stop, stop)
            crates = data["demands"][data["pod_names"].index(stop)]
            parts.append(f"{i+1}-{real} ({crates})")
        route_str = " → ".join(parts)

        output.append({
            "truck":     v["name"],
            "trip":      v.get("trip", 1),
            "label":     trip["label"],     # ← now included
            "type":      v["type"],
            "route":     route_str,
            "load":      trip["load"],
            "distance":  round(trip["distance"], 2),
            "fuel_cost": round(fuel_cost, 2),
            "rent_cost": rent_cost,
            "total_cost":total,
        })

    df_out = pd.DataFrame(output)
    st.session_state.update({
        'vrp_df_output':   df_out,
        'vrp_trips':       trips,
        'vrp_output_data': output,
        'vrp_trip_labels':[o['label'] for o in output],  # now safe
        'vrp_data':        data
    })

# --- DISPLAY RESULTS ---
if st.session_state.get('vrp_df_output') is not None and not st.session_state['vrp_df_output'].empty:
    st.header("Optimized Trips")
    st.dataframe(st.session_state['vrp_df_output'])
    sel = st.selectbox("Pick a route:", st.session_state['vrp_trip_labels'])
    idx = st.session_state['vrp_trip_labels'].index(sel)
    trip = st.session_state['vrp_trips'][idx]

    stops = [depot] + trip["route"] + [depot]
    coords = [coords_dict[data["pod_mapping"].get(pt,pt)] for pt in stops]
    m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
    try:
        geo = ors_client.directions(coordinates=coords, profile='driving-hgv', format='geojson')['features'][0]['geometry']
        folium.GeoJson(geo, style_function=lambda x:{'color':'blue','weight':5}).add_to(m)
    except:
        pass
    for pt,c in zip(stops, coords):
        folium.CircleMarker(c[::-1], radius=7, popup=pt).add_to(m)
    st.subheader(sel)
    st_folium(m, width=750, height=500)
else:
    st.info("Enter demands & click Optimize.")
