import torch
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from skills import GNNModel, train_model

def run_academicinterest():
    student_data = pd.read_csv('student_data.csv',nrows=31)
    G_academic = nx.Graph()
    # Function to add edges based on shared academic interests
    def add_nodes_and_edges_based_on_academic_interests(G):
        for i, student1 in student_data.iterrows():
            G.add_node(student1['StudentID'], academic_interest=student1['AcademicInterest'])
            for j, student2 in student_data.iterrows():
                if i != j:
                    interest1 = student1['AcademicInterest']
                    interest2 = student2['AcademicInterest']
                    if interest1 == interest2:
                        G.add_edge(student1['StudentID'], student2['StudentID'], weight=1)


    # Add edges based on academic interests
    add_nodes_and_edges_based_on_academic_interests(G_academic)

    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_academic)
    nx.draw(G_academic, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=1000)
    plt.title('Student Collaboration Graph Based on Shared Academic Interests')
    plt.show()
    # Convert graph to PyTorch Geometric Data object
    edge_index = torch.tensor([[list(G_academic.nodes).index(n1), list(G_academic.nodes).index(n2)] for n1, n2 in G_academic.edges()], dtype=torch.long).t().contiguous()
    x = torch.tensor([[node] for node in range(len(G_academic.nodes))], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    model = GNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_model(model, data, optimizer, criterion)
    # Define the recommendation function based on academic interests
    def recommend_collaboration_partners_by_interests(student_id, student_data, G, model, threshold=-1):
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
            neighbor_academic = neighbor_info['AcademicInterest'].iloc[0]
            prediction = model(data).detach().numpy()[neighbor_id]
            print("Prediction:", prediction)
            if prediction > threshold:
                collaboration_partners.append({'StudentID': neighbor_id, 'Name': neighbor_name, 'AcademicInterest': neighbor_academic})
        return collaboration_partners

# Input student ID
    student_id = int(input("Enter your Student ID: "))

# Recommend collaboration partners based on academic interests
    partners_by_interests = recommend_collaboration_partners_by_interests(student_id, student_data,G_academic, model)

    print("Recommended collaboration partners based on your academic interests:")
    for partner in partners_by_interests:
        print(partner)
