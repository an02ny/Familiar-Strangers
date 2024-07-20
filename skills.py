import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
# Load only one sample from the dataset
# Assuming you have a CSV file named 'student_data.csv' with the provided attributes
def run_skills():
    student_data = pd.read_csv('student_data.csv',nrows=31)

    # Construct a graph
    G = nx.Graph()

    # Function to add edges based on shared skills
    def add_edges_based_on_skills(G):
        for i, student1 in student_data.iterrows():
            G.add_node(student1['StudentID'], skills=student1['Skills'])
            for j, student2 in student_data.iterrows():
                if i != j:
                    skills1 = set(student1['Skills'].split(', '))
                    skills2 = set(student2['Skills'].split(', '))
                    #print(skills1.intersection(skills2))
                    common_skills = len(skills1.intersection(skills2))
                    if common_skills > 0:
                        G.add_edge(i,j,weight=common_skills)

    add_edges_based_on_skills(G)
    # Visualize the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=1000)
    plt.title('Student Collaboration Graph Based on Shared Skills')
    plt.show()

    # Convert graph to PyTorch Geometric Data object
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x = torch.tensor([[node[0]] for node in G.nodes(data=True)], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    # Define the Graph Neural Network model
    class GNNModel(nn.Module):
        def __init__(self):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(1, 16)
            self.conv2 = GCNConv(16, 1)
            self.dropout = nn.Dropout(p=0.5)  # Dropout layer for regularization 
            
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # Train the model
    def train_model(model, data, optimizer, criterion, num_epochs=200):
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.ones_like(output))  # Dummy target
            loss.backward()
            optimizer.step()

    # Function to recommend collaboration partners based on skills
    def recommend_collaboration_partners_by_skills(student_id, student_data,G ,threshold=0.5):
        model=GNNModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        train_model(model, data, optimizer, criterion)
        student_neighbors = list(G.neighbors(student_id))
        collaboration_partners = []
        for neighbor_id in student_neighbors:
            if neighbor_id < len(student_data):
                neighbor_info = student_data.loc[student_data['StudentID'] == neighbor_id]
                neighbor_name = neighbor_info['Name'].iloc[0]
                neighbor_skills = neighbor_info['Skills'].iloc[0]
                prediction = model(data).detach().numpy()[neighbor_id]
                if prediction > threshold:
                    print("Prediction:", prediction)
                    collaboration_partners.append({'StudentID': neighbor_id, 'Name': neighbor_name, 'Skills': neighbor_skills})
        return collaboration_partners
    
# Input student ID
    student_id = int(input("Enter your Student ID: "))

#Recommend collaboration partners based on skills
    partners = recommend_collaboration_partners_by_skills(student_id, student_data,G,threshold=0.4)

    print("Recommended collaboration partners based on your skills:")
    for partner in partners:
        print(partner)
class GNNModel(nn.Module):
        def __init__(self):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(1, 16)
            self.conv2 = GCNConv(16, 1)
            self.dropout = nn.Dropout(p=0.5)  # Dropout layer for regularization 
            
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x

def train_model(model, data, optimizer, criterion, num_epochs=200):
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.ones_like(output))  # Dummy target
            loss.backward()
            optimizer.step()