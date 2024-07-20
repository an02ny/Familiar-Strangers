import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from skills import GNNModel, train_model

def run_clubmembership():
    student_data = pd.read_csv('student_data.csv',nrows=31)
    G_club = nx.Graph()
    # Function to add edges based on shared academic interests
    def add_nodes_and_edges_based_on_ClubMemberships(G):
        for i, student1 in student_data.iterrows():
            G.add_node(student1['StudentID'], club_memeberships=student1['ClubMemberships'])
            for j, student2 in student_data.iterrows():
                if i != j:
                    interest1 = set(student1['ClubMemberships'].split(', '))
                    interest2 = set(student2['ClubMemberships'].split(', '))
                    common=len(interest1.intersection(interest2))
                    if common>0:
                        G.add_edge(student1['StudentID'], student2['StudentID'], weight=common)


    # Add edges based on academic interests
    add_nodes_and_edges_based_on_ClubMemberships(G_club)

    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_club)
    nx.draw(G_club, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=1000)
    plt.title('Student Collaboration Graph Based on ClubMemberships')
    plt.show()

    # Convert graph to PyTorch Geometric Data object
    edge_index = torch.tensor([[list(G_club.nodes).index(n1), list(G_club.nodes).index(n2)] for n1, n2 in G_club.edges()], dtype=torch.long).t().contiguous()
    x = torch.tensor([[node] for node in range(len(G_club.nodes))], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # Define the recommendation function based on academic interests
    def recommend_collaboration_partners_by_ClubMemberships(student_id, student_data,G, threshold=0.5):
        model = GNNModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_model(model, data, optimizer, criterion)
        
        student_neighbors = list(G.neighbors(student_id))
        collaboration_partners = []
        for neighbor_id in student_neighbors:
            #print("Neighbor ID:", neighbor_id)
            print(neighbor_id)
            if neighbor_id+1 < len(student_data):
                neighbor_info = student_data.loc[student_data['StudentID'] == neighbor_id]
                neighbor_name = neighbor_info['Name'].iloc[0]
                neighbor_clubs = neighbor_info['ClubMemberships'].iloc[0]
                prediction = model(data).detach().numpy()[neighbor_id]
                print("Prediction:", prediction)
                if prediction > threshold:
                    collaboration_partners.append({'StudentID': neighbor_id, 'Name': neighbor_name, 'ClubMemberships': neighbor_clubs})
        return collaboration_partners

# Input student ID
    student_id = int(input("Enter your Student ID: "))

# Recommend collaboration partners based on academic interests
    partners_by_interests = recommend_collaboration_partners_by_ClubMemberships(student_id, student_data,G_club)

    print("Recommended collaboration partners based on your ClubMemberships:")
    for partner in partners_by_interests:
        print(partner)
