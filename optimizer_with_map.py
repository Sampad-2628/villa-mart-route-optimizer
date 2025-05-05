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
    "Kanan Vihar - RTO": [85.82865, 20.36067],
    "Sum Hospital":       [85.77249, 20.28318],
    "Symphony Mall":      [85.88286, 20.32356],
    "Aiims":              [85.78570, 20.24993],
    "Bapuji Nagar":       [85.84149, 20.25109],
    "Baramunda":          [85.80296, 20.28139],
    "District Centre (CSPUR)": [85.81827, 20.32664],
    "Patia":              [85.82528, 20.35302],
    "Kalinga":            [85.81536, 20.29597],
    "Bomikhal":           [85.85578, 20.28113],
    "PatraPada":          [85.76613, 20.23536],    # ← Newly added
    "Villamart (Depot)":  [85.77015, 20.24394],
}

ORS_API_KEY = '5b3ce3597851110001cf62488858392856e24062ae6ba005c2e38325'
ors_client = openrouteservice.Client(key=ORS_API_KEY)

st.title("Villa Mart Route Optimizer (Smart Pod-Splitting, Cost & Utilization)")

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
own_truck_b = {"name": "Truck B", "capacity": 90,  "max_trips": 2}
own_trucks = [own_truck_a, own_truck_b]
rent_capacity = st.number_input("Rented truck capacity (crates)", min_value=70, max_value=300, value=121)
max_rented_trips = st.number_input("Max rented trips allowed", min_value=0, max_value=10, value=5)
avg_speed = st.number_input("Average truck speed (km/hr)", min_value=30, max_value=100, value=60)
petrol_price = st.number_input("Current petrol price (₹/litre)", min_value=0, max_value=1000, value=101)
mileage = 12.0  # fixed for all trucks

run_optim = st.button("Optimize Route")

def create_data_model():
    max_truck_capacity = max([t["capacity"] for t in own_trucks] + [rent_capacity])
    pod_expanded, pod_mapping, pod_demands, pod_coords = [], {}, [], []
    for store in store_list[:-1]:
        demand = user_crate_demand[store]
        if demand > max_truck_capacity:
            n_full = demand // max_truck_capacity
            last   = demand % max_truck_capacity
            for i in range(n_full):
                name = f"{store}__part{i+1}"
                pod_expanded.append(name); pod_mapping[name]=store
                pod_demands.append(max_truck_capacity); pod_coords.append(coords_dict[store])
            if last > 0:
                name = f"{store}__part{n_full+1}"
                pod_expanded.append(name); pod_mapping[name]=store
                pod_demands.append(last); pod_coords.append(coords_dict[store])
        else:
            pod_expanded.append(store); pod_mapping[store]=store
            pod_demands.append(demand); pod_coords.append(coords_dict[store])

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

    # Vehicle setup
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
        "vehicle_capacities":[v["capacity"] for v in vehicles],
        "num_vehicles": len(vehicles),
        "depot": 0,
        "vehicles": vehicles,
        "pod_names": all_points,
        "display_names": [depot]+pod_expanded,
        "pod_mapping": pod_mapping,
        "coords": all_coords,
        "truck_trip_labels": labels,
        "petrol_price": petrol_price,
        "mileage": mileage,
        "km_matrix": expanded_dist,
    }

def solve_vrp(data):
    mgr = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(mgr)

    # Optional fixed cost on rented vehicles
    for vid,v in enumerate(data["vehicles"]):
        if v["type"]=="rented":
            routing.SetFixedCostOfVehicle(100000, vid)

    def dist_cb(i,j):
        return int(data["distance_matrix"][mgr.IndexToNode(i)][mgr.IndexToNode(j)] * 100)
    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_cb(i):
        return data["demands"][mgr.IndexToNode(i)]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, data["vehicle_capacities"], True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.seconds = 30
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = routing.SolveWithParameters(params)
    trips = []
    if sol:
        for vid in range(data["num_vehicles"]):
            idx = routing.Start(vid)
            route, load, dist_km = [], 0, 0
            while not routing.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node!=0:
                    route.append(data["pod_names"][node])
                load += data["demands"][node]
                prev = idx
                idx = sol.Value(routing.NextVar(idx))
                if not routing.IsEnd(idx):
                    dist_km += data["km_matrix"][mgr.IndexToNode(prev)][mgr.IndexToNode(idx)]
            if route:
                trips.append({
                    "vehicle_id": vid,
                    "vehicle": data["vehicles"][vid],
                    "route": route,
                    "load": load,
                    "distance": dist_km,
                    "label": data["truck_trip_labels"][vid]
                })
    return trips

if run_optim:
    data  = create_data_model()
    trips = solve_vrp(data)
    output=[]
    for trip in trips:
        v = trip["vehicle"]
        is_r  = (v["type"]=="rented")
        rent_cost = 2300 if is_r and trip["distance"]>30 else (1500 if is_r else 0)
        fuel_cost = ((trip["distance"]/data["mileage"])*data["petrol_price"]) if not is_r else 0
        total = fuel_cost + rent_cost

        # Build human-friendly route string
        parts=[]
        for idx,stop in enumerate(trip["route"]):
            real = data["pod_mapping"].get(stop, stop)
            crates = data["demands"][data["pod_names"].index(stop)]
            tag = f"{real} ({crates})" if stop==real else f"{real} split ({crates})"
            parts.append(f"{idx+1}-{tag}")
        route_str=" -> ".join(parts)

        output.append({
            "truck": v["name"],
            "trip":  v.get("trip",1),
            "type":  v["type"],
            "route": route_str,
            "load":  trip["load"],
            "distance": round(trip["distance"],2),
            "fuel_cost": round(fuel_cost,2),
            "rent_cost": rent_cost,
            "total_cost": round(total,2),
        })

    df_out = pd.DataFrame(output)
    st.session_state.update({
        'vrp_df_output':  df_out,
        'vrp_trips':      trips,
        'vrp_output_data':output,
        'vrp_trip_labels':[o['label'] for o in output],
        'vrp_data':       data
    })

# --- DISPLAY RESULTS ---
if st.session_state.get('vrp_df_output', pd.DataFrame()).shape[0]>0:
    st.header("Optimized Trip Plan (Smart Split)")
    st.dataframe(st.session_state['vrp_df_output'])
    sel = st.selectbox("Show route for:", st.session_state['vrp_trip_labels'])
    idx = st.session_state['vrp_trip_labels'].index(sel)
    trip = st.session_state['vrp_trips'][idx]
    stops = [depot]+trip["route"]+[depot]
    m = folium.Map(location=coords_dict[depot][::-1], zoom_start=12)

    coords = [coords_dict[data["pod_mapping"].get(pt,pt)] for pt in stops]
    try:
        geo = ors_client.directions(coordinates=coords, profile='driving-hgv', format='geojson')['features'][0]['geometry']
        folium.GeoJson(geo, style_function=lambda x:{'color':'blue','weight':6,'opacity':0.85}).add_to(m)
    except Exception as ex:
        st.warning(f"Couldn’t plot real route: {ex}")

    for pt,c in zip(stops, coords):
        folium.CircleMarker(location=c[::-1], radius=8, popup=pt).add_to(m)
    st.subheader(f"Route Map: {sel}")
    st_folium(m, width=800, height=600)
else:
    st.info("Enter demands & click Optimize to see results.")
