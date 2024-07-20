import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from skills import GNNModel, train_model

# Load student data from CSV
def run_extracurricularactivities():
    student_data = pd.read_csv('student_data.csv', nrows=31)

    # Create a graph
    G_extracurricular = nx.Graph()

    # Function to add nodes and edges based on shared extracurricular activities
    def add_nodes_and_edges_based_on_activities(G):
        for i, student1 in student_data.iterrows():
            G.add_node(student1['StudentID'], extracurricular_activities=student1['ExtracurricularActivities'])
            for j, student2 in student_data.iterrows():
                if i != j:
                    activity1 = student1['ExtracurricularActivities']
                    activity2 = student2['ExtracurricularActivities']
                    if activity1 == activity2:
                        G.add_edge(student1['StudentID'], student2['StudentID'], weight=1)

    # Add nodes and edges based on extracurricular activities
    add_nodes_and_edges_based_on_activities(G_extracurricular)

    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_extracurricular)
    nx.draw(G_extracurricular, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=100)
    plt.title('Student Collaboration Graph Based on Shared Extracurricular Activities')
    plt.show()

    # Convert graph to PyTorch Geometric Data object
    edge_index = torch.tensor([[list(G_extracurricular.nodes).index(n1), list(G_extracurricular.nodes).index(n2)] for n1, n2 in G_extracurricular.edges()], dtype=torch.long).t().contiguous()
    x = torch.tensor([[node] for node in range(len(G_extracurricular.nodes))], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # Train the model   
    model = GNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_model(model, data, optimizer, criterion)

    # Define the recommendation function based on extracurricular activities
    def recommend_collaboration_partners_by_activities(student_id, student_data, G, model, threshold=-1):
        model = GNNModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_model(model, data, optimizer, criterion)
    
        student_neighbors = list(G.neighbors(student_id))
        collaboration_partners = []
        for neighbor_id in student_neighbors:
            print("Neighbor ID:", neighbor_id)
            neighbor_info = student_data.loc[student_data['StudentID'] == neighbor_id]
            neighbor_name = neighbor_info['Name'].iloc[0]
            neighbor_extra = neighbor_info['ExtracurricularActivities'].iloc[0]
            prediction = model(data).detach().numpy()[neighbor_id]
            print("Prediction:", prediction)
            if prediction > threshold:
                collaboration_partners.append({'StudentID': neighbor_id, 'Name': neighbor_name, 'ExtracurricularActivities': neighbor_extra})
        return collaboration_partners


# Input student ID
    student_id = int(input("Enter your Student ID: "))

# Recommend collaboration partners based on extracurricular activities
    partners_by_activities = recommend_collaboration_partners_by_activities(student_id, student_data, G_extracurricular, model)

    print("Recommended collaboration partners based on your extracurricular activities:")
    for partner in partners_by_activities:
        print(partner)
