#!/usr/bin/env python
# coding: utf-8

# Task-I

# In[2]:


import pandas as pd
from geopy.geocoders import Nominatim

common_path = '/Users/prabhakarmanikyam/Documents/ISE 535/Project/Project 1/Project1_Team20_Submission/'

csv_file_path = common_path +'Pythonproject_Dallasdata.csv'  
cd = pd.read_csv(csv_file_path)

# Use nominatim to initialize a geolocator
geolocator = Nominatim(user_agent="address_geocoder", timeout = 10)

# Define a function to geocode an address and apply geocoding to each address and create new columns for latitude and longitude
def geocode_address(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
cd['Latitude'], cd['Longitude'] = zip(*cd['Address'].apply(geocode_address))


output_csv_file_path = common_path + 'Pythonproject_DallasdataCo-ordinates.csv'
cd.to_csv(output_csv_file_path, index=False)

print(f'Geocoded data saved to the location {output_csv_file_path}')


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import random

#Setting a RandomSeed for Service Station reproducibilty 
random.seed(42)


df = pd.read_csv(common_path+'Pythonproject_DallasdataCo-ordinates.csv')

# Set the boundaries for cloud kitchens
min_latitude = df['Latitude'].min() - 0.036
min_longitude = df['Longitude'].min() - 0.045
max_latitude = df['Latitude'].max() + 0.036
max_longitude = df['Longitude'].max() + 0.045

# Creating a Scatterplot Map for the Cloud Kitchens
plt.figure(figsize=(10, 10))
plt.scatter(df['Longitude'], df['Latitude'], c='red', label='Cloud_Kitchens')
plt.title('Cloud_Kitchens and Service_Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Creating 50 service station points at random within the boundaries
sampled_pts = []
for _ in range(50):
    rand_latitude = random.uniform(min_latitude, max_latitude)
    rand_longitude = random.uniform(min_longitude, max_longitude)
    sampled_pts.append((rand_latitude, rand_longitude))

sampled_data = pd.DataFrame({'Latitude': [point[0] for point in sampled_pts], 'Longitude': [point[1] for point in sampled_pts]})

sampled_data.to_csv(common_path +'random_points.csv', index=False)

sampled_df = pd.DataFrame({'Latitude': sampled_data['Latitude'], 'Longitude': sampled_data['Longitude']})
plt.scatter(sampled_df['Longitude'], sampled_df['Latitude'], c='blue', label='Sampled Service Stations')
plt.legend(loc="upper left")

plt.show()
#Deliverable 3
plt.savefig('Locations.jpg')


# In[4]:


import pandas as pd
from tabulate import tabulate

#Tabulating the result into 3 columns address, Zipcode and Co-ordinates 
data = pd.read_csv(common_path + 'Pythonproject_DallasdataCo-ordinates.csv')


data.index = [f'c{i+1}' for i in range(25)]

data['Zipcode'] = data['Address'].str.extract(r'(\d{5})')
data['Address'] = data['Address'].str.replace(r'\d{5}', '').str.strip()


data['Co-ordinates'] = list(zip(data['Latitude'], data['Longitude']))


data = data[['Address', 'Zipcode', 'Co-ordinates']]

table = tabulate(data, headers='keys', tablefmt='simple')
print(table)

# Deliverable 2
data.to_csv(common_path + 'Locations.txt', index=False,sep='\t')


# In[5]:


import pandas as pd
from tabulate import tabulate
# Creating and tabulating new data of service station locations for later use 
data = pd.read_csv(common_path + 'random_points.csv')

data.index = [f'S{i+1}' for i in range(50)]

data['Location'] = list(zip(data['Latitude'], data['Longitude']))

data = data[['Location']]

table = tabulate(data, headers='keys', tablefmt='grid')
print(table)

data.to_csv(common_path + 'service_stations_co-ordinates.txt', index=False,sep='\t')


# In[6]:


import pandas as pd
import math
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    
    # Used mean radius of earth
    radius = 3958.8  

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance

def calculate_distances(cloud_kitchens_file, service_stations_file):
#Calculate distances between cloud kitchens and service stations as Distance matrix [dij]
    
    cloud_kitchens_data = pd.read_csv(cloud_kitchens_file)
    service_stations_data = pd.read_csv(service_stations_file)

    dij = []

    for _, kitchen in cloud_kitchens_data.iterrows():
        row = []
        for _, station in service_stations_data.iterrows():
            distance = haversine_distance(kitchen['Latitude'], kitchen['Longitude'], station['Latitude'], station['Longitude'])
            row.append(distance)
        dij.append(row)

    return dij

cloud_kitchens_file = common_path + 'Pythonproject_DallasdataCo-ordinates.csv'
service_stations_file = common_path + 'random_points.csv'

dij = calculate_distances(cloud_kitchens_file, service_stations_file)

print(dij)


# In[7]:


import pandas as pd
# Create a DataFrame from dij distance matrix
df = pd.DataFrame(dij)

# Deliverable 4
df.to_csv(common_path + 'Distances.csv', index=False, header=False)


# Task-II

# In[8]:


import pulp
import pandas as pd



dij_df = pd.read_csv(common_path + 'Distances.csv', header=None)

# Define sets I and J based on the dij matrix
I = range(len(dij_df))
J = range(len(dij_df.columns))

D = {}
for i in range(len(dij)):
    D[i] = {j: dij[i][j] for j in range(len(dij[i]))}

# Now we create a linear programming model
model = pulp.LpProblem("Assignment_Problem", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(25) for j in range(50)), lowBound=0, upBound=1, cat='Integer')

# Define objective function for minimization, this case we want to minimize distance
model += (pulp.lpSum([D[i][j] * x[(i, j)] for i in range(25) for j in range(50)]), "Total Distance")

# Constraint 1: Each service station is served by exactly 1 kitchen
for j in range(50):
    model += (pulp.lpSum([x[(i, j)] for i in range(25)]) == 1, f"Constraint1{j}")

# Constraint 2: Each kitchen serves exactly 2 service stations
for i in range(25):
    model += (pulp.lpSum([x[(i, j)] for j in range(50)]) == 2, f"Constraint2{i}")



# We solve our problem and print the results
model.solve()
status = pulp.LpStatus[model.status]

result = []    # list to store the "Origin and Destination (OD)" data

for i in range(25):
    for j in range(50):
        if x[(i, j)].varValue == 1:
            result.append([i, j, D[i][j]])    # append pair of Kitchen and Service Station with their distance
                 # Uncomment next line to debug 
            print(f"Kitchen {i} is assigned to Service Station {j}")
    
#Deliverable 5
model.writeLP("AP.mps")




# In[12]:


import scipy.sparse
#Deliverable 6
sparse_matrix = scipy.sparse.coo_matrix([[D[i][j] if x[(i, j)].varValue == 1 else 0 for j in range(50)] for i in range(25)])

result = []  

for i in range(25):
    for j in range(50):
        if x[(i, j)].varValue == 1:
            result.append([i, j, D[i][j]])  
print(result)

sparse_matrix_csr = sparse_matrix.tocsr()


csr_file_path = common_path + "Solution.csr"


with open(csr_file_path, "w") as csr_file:
    
#Saved as solution.csr file
    for row in range(sparse_matrix_csr.shape[0]):
        csr_file.write(" ".join(map(str, sparse_matrix_csr.indices[sparse_matrix_csr.indptr[row]:sparse_matrix_csr.indptr[row + 1]])))
        csr_file.write("\n")

print(f"Saved the CSR matrix to {csr_file_path}")


# In[13]:



import pandas as pd
#Building and tabulating as origin and destination (OD)
result_tbl = pd.DataFrame(result, columns=["Cloud_Kitchen_Index(origin)", "Service_Station_Index(destination)", "Distance_in_miles"])
headers = ["Cloud_Kitchen_Index(origin)", "Service_Station_Index(destination)", "Distance_in_miles"]
table = tabulate(result_tbl, headers=headers, tablefmt="simple")
print(table)
#Deliverable 7
result_tbl.to_csv(common_path + 'OD.txt', sep='\t', index=False)


# Task-III

# In[16]:


import matplotlib.pyplot as plt


#Creating a frequency graph using for three different distance ranges Short(s),Medium(m),Large(l)
# Defining the distance ranges
s_range = (0, 3)
m_range = (3, 6)
l_range = (6, float('inf'))

s_count = 0
m_count = 0
l_count = 0

distances = result_tbl["Distance_in_miles"]

# Categorizing distances into ranges
for distance in distances:
    if s_range[0] <= distance < s_range[1]:
        s_count += 1
    elif m_range[0] <= distance < m_range[1]:
        m_count += 1
    elif distance >= l_range[0]:
        l_count += 1

t_assignments = s_count + m_count + l_count


short_percentage = (s_count / t_assignments) * 100
medium_percentage = (m_count / t_assignments) * 100
long_percentage = (l_count / t_assignments) * 100

# plotting bar graph
distance_ranges = ['< 3 miles', '3-6 miles', '> 6 miles']
percentages = [short_percentage, medium_percentage, long_percentage]

plt.bar(distance_ranges, percentages, color=['green', 'blue', 'red'])
plt.xlabel('Distance Ranges')
plt.ylabel('Frequency (%)')
plt.title('Frequency of Assignments by Distance Range')
plt.ylim(0, 100)  # Set the y-axis limit to 0-100%
plt.show()
#Deliverable 8
plt.savefig('Frequency.jpeg',format="jpeg")


# In[17]:


import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Adding nodes for Cloud Kitchens and Service Stations
for cloud_kitchen_index in I:
    G.add_node(f'CKitchen{cloud_kitchen_index}', type='CKitchen')
for service_station_index in J:
    G.add_node(f'SStation{service_station_index}', type='SStation')

# Adding and defining edges from Cloud Kitchens to their assigned Service Stations
for cloud_kitchen_index in I:
    service_station_1 = result[cloud_kitchen_index][0]
    service_station_2 = result[cloud_kitchen_index][1]
    
    G.add_edge(f'CKitchen{cloud_kitchen_index}', f'SStation{service_station_1}', label=f'CKitchen{cloud_kitchen_index} to SStation{service_station_1}')
    G.add_edge(f'CKitchen{cloud_kitchen_index}', f'SStation{service_station_2}', label=f'CKitchen{cloud_kitchen_index} to SStation{service_station_2}')

edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}

# Separate nodes by type
cloud_kitchens = [node for node in G.nodes if G.nodes[node]['type'] == 'CKitchen']
service_stations = [node for node in G.nodes if G.nodes[node]['type'] == 'SStation']

plt.figure(figsize=(12, 6))
pos = nx.spring_layout(G, seed=42)

# plotting Cloud Kitchens and service stations
nx.draw_networkx_nodes(G, pos, nodelist=cloud_kitchens, node_size=75, node_color='red', node_shape='o', label='Cloud Kitchens')

nx.draw_networkx_nodes(G, pos, nodelist=service_stations, node_size=75, node_color='green', node_shape='s', label='Service Stations')

nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=5) 

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, font_color='black')

plt.title("Assignment Solution", fontweight='bold')

# Deliverable 9
plt.savefig('Solution.jpeg')
plt.show()


# In[ ]:




