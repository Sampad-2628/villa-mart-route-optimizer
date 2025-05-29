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

ORS_API_KEY = '5b3ce3597851110001cf62488858392856e24062ae6ba005c2e38325'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Villa Mart Route Optimizer (Smart Split + Timing)")

# --- INPUTS ---
st.header("1. Crate Demand")
user_crate_demand = {}
total_crates = 0
for store in store_list[:-1]:
    val = st.number_input(f"{store}", min_value=0, max_value=3000, value=10)  # Increased max_value for higher demands
    user_crate_demand[store] = val
    total_crates += val
st.write(f"**Total crates:** {total_crates}")

st.header("2. Truck & Cost Settings")
own_truck_a = {"name": "Truck A", "capacity": 121, "max_trips": 2}
own_truck_b = {"name": "Truck B", "capacity": 90,  "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity    = st.number_input("Rented truck capacity (crates)", 70, 300, 121)
max_rented_trips = st.number_input("Max rented trips allowed", 0, 20, 10)
petrol_price     = st.number_input("Petrol price (₹/litre)", 0, 1000, 101)
mileage          = 12.0  # fixed for all trucks
avg_speed        = st.number_input("Average truck speed (km/hr)", 30, 200, 70)

run_optim = st.button("Optimize Route")

# --- DATA MODEL & SOLVER ---
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
                pod_demands.append(rem);    pod_coords.append(coords_dict[store])
        else:
            pod_expanded.append(store); pod_mapping[store]=store
            pod_demands.append(d);       pod_coords.append(coords_dict[store])

    all_pts = [depot] + pod_expanded
    km_mat  = []
    for src in all_pts:
        src_actual = pod_mapping.get(src, src)
        row = [df_distance.loc[src_actual, pod_mapping.get(dst, dst)] for dst in all_pts]
        km_mat.append(row)

    # --- Improved Dynamic Truck Rounds Logic ---
    vehicles, labels = [], []
    demand_left = sum(user_crate_demand.values())

    # Fill owned trucks first (all rounds)
    for t in own_trucks:
        for trip in range(t["max_trips"]):
            if demand_left <= 0:
                break
            vehicles.append({
                "type": "own",
                "name": t["name"],
                "capacity": t["capacity"],
                "trip": trip + 1
            })
            labels.append(f"{t['name']} - Trip {trip+1}")
            demand_left -= t["capacity"]

    # Fill with rented trucks (one trip per rental) if needed
    rented_needed = 0
    while demand_left > 0 and rented_needed < max_rented_trips:
        vehicles.append({
            "type": "rented",
            "name": f"Rented-{rented_needed+1}",
            "capacity": rent_capacity,
            "trip": 1
        })
        labels.append(f"Rented-{rented_needed+1}")
        demand_left -= rent_capacity
        rented_needed += 1

    # Warn if demand still left after all rounds/trucks
    if demand_left > 0:
        st.warning(
            f"Warning: Even after using all available truck rounds, {demand_left} crates remain unplanned. "
            "Increase 'Max rented trips allowed' or truck capacities to cover all demand."
        )

    return {
        "distance_matrix": km_mat,
        "demands":          [0] + pod_demands,
        "vehicle_capacities": [v["capacity"] for v in vehicles],
        "num_vehicles":     len(vehicles),
        "depot":            0,
        "vehicles":         vehicles,
        "pod_names":        all_pts,
        "pod_mapping":      pod_mapping,
        "truck_trip_labels":labels,
        "petrol_price":     petrol_price,
        "mileage":          mileage,
        "avg_speed":        avg_speed,
        "km_matrix":        km_mat,
    }

def solve_vrp(data):
    mgr    = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing= pywrapcp.RoutingModel(mgr)
    for vid, v in enumerate(data["vehicles"]):
        if v["type"]=="rented":
            routing.SetFixedCostOfVehicle(100000, vid)

    def dist_cb(i,j):
        return int(data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)] * 100)
    t_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(t_idx)

    def dem_cb(i):
        return data["demands"][mgr.IndexToNode(i)]
    d_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(d_idx, 0, data["vehicle_capacities"], True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 30
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = routing.SolveWithParameters(params)
    trips = []
    if sol:
        for vid in range(data["num_vehicles"]):
            idx, route, load = routing.Start(vid), [], 0
            while not routing.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node!=0:
                    route.append(data["pod_names"][node])
                load += data["demands"][node]
                prev, idx = idx, sol.Value(routing.NextVar(idx))
            if route:
                trips.append({
                    "vehicle_id": vid,
                    "vehicle":     data["vehicles"][vid],
                    "route":       route,
                    "load":        load,
                    "label":       data["truck_trip_labels"][vid]
                })
    return trips

def format_time(mins):
    h = int(mins//60); m = int(mins%60)
    return f"{h}h {m}m" if h else f"{m}m"

# --- RUN + BUILD OUTPUT ---
if run_optim:
    data   = create_data_model()
    trips  = solve_vrp(data)
    output = []
    for trip in trips:
        # compute total km including return
        seq = [depot] + trip["route"] + [depot]
        total_km = sum(
            data["km_matrix"][data["pod_names"].index(seq[i])][data["pod_names"].index(seq[i+1])]
            for i in range(len(seq)-1)
        )
        # timing
        travel_min  = total_km / data["avg_speed"] * 60
        service_min = len(trip["route"]) * 15 + trip["load"] * 0.5
        total_min   = travel_min + service_min
        # cost
        v = trip["vehicle"]
        is_r = (v["type"]=="rented")
        rent_cost = 2300 if is_r and total_km>30 else (1500 if is_r else 0)
        fuel_cost = (total_km/data["mileage"]) * data["petrol_price"] if not is_r else 0
        # build route string
        parts = [
            f"{i+1}-{data['pod_mapping'].get(stop,stop)} ({data['demands'][data['pod_names'].index(stop)]})"
            for i,stop in enumerate(trip["route"])
        ]
        route_str = " → ".join(parts)
        # collect
        output.append({
            "truck":       v["name"],
            "trip":        trip.get("trip",1),
            "label":       trip["label"],
            "route":       route_str,
            "load":        trip["load"],
            "distance_km": round(total_km,2),
            "travel_time": format_time(travel_min),
            "service_time":format_time(service_min),
            "total_time":  format_time(total_min),
            "fuel_cost":   round(fuel_cost,2),
            "rent_cost":   rent_cost,
            "total_cost":  round(rent_cost+fuel_cost,2),
        })
    df_out = pd.DataFrame(output)
    st.session_state['vrp_df_output']   = df_out
    st.session_state['vrp_trips']       = trips
    st.session_state['vrp_trip_labels'] = [o['label'] for o in output]
    st.session_state['vrp_data']        = data

# --- DISPLAY RESULTS ---
if 'vrp_df_output' in st.session_state and not st.session_state['vrp_df_output'].empty:
    df_output   = st.session_state['vrp_df_output']
    trips       = st.session_state['vrp_trips']
    labels      = st.session_state['vrp_trip_labels']
    data        = st.session_state['vrp_data']

    st.header("Optimized Trip Plan (with Timing)")
    st.dataframe(df_output)

    sel = st.selectbox("Select truck/trip:", labels)
    idx = labels.index(sel)
    trip = trips[idx]

    seq = [depot] + trip["route"] + [depot]
    coords = [coords_dict[data["pod_mapping"].get(pt,pt)] for pt in seq]

    m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)
    try:
        geo = ors_client.directions(coordinates=coords, profile='driving-hgv', format='geojson')['features'][0]['geometry']
        folium.GeoJson(geo, style_function=lambda x:{'color':'blue','weight':6,'opacity':0.85}).add_to(m)
    except Exception as e:
        st.warning(f"Route plotting failed: {e}")

    for pt, c in zip(seq, coords):
        if pt==depot:
            lbl = f"{depot} (Depot, 0 crates)"
        else:
            idx_pt = data["pod_names"].index(pt)
            lbl = f"{data['pod_mapping'].get(pt,pt)}: {data['demands'][idx_pt]} crates"
        folium.CircleMarker(location=c[::-1], radius=8, popup=lbl, tooltip=lbl).add_to(m)

    st.subheader(f"Route Visualization: {sel}")
    st_folium(m, width=800, height=600)
else:
    st.info("Enter demands & click Optimize to run.")
