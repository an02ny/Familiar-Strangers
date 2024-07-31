# Familiar Strangers
_Uncover the hidden connections with like-minded but unspoken people in our daily lives!_



Familiar Strangers is a project that uses **Social Network Analysis** and **Graph Neural Networks** to identify shared skills and interests among  500+ university students.
The goal is to facilitate cross-disciplinary collaborations and project partnerships, demonstrating the potential of machine learning and graph analysis techniques in educational settings.

This repository contains scripts and data for analyzing student information, including academic interests, club memberships, extracurricular activities, and research projects. The project aims to provide insights into students' academic and extracurricular engagement.

## Features
- **Graph Construction**: Create a graph based on shared skills and interests among students.
- **Visualization**: Display the graph of student collaborations.
- **Graph Neural Network (GNN)**: Train a GNN model to recommend collaboration partners.
- **Multiple Analysis Options**: Analyze skills, research, major selections, extracurricular activities, club memberships, and academic interests.


## Set up

1. Clone the repository:
    ```bash
    git clone https://github.com/an02ny/Familiar_Strangers.git
    cd Familiar_Strangers
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.Run the main script
Execute the main.py script to start the program:
   
   ```bash
   python3 main.py
 ```
   
4.Follow the prompts to select the type of analysis you want to perform.

## Contents
- `student_data.csv`: Contains the student data used for analysis.
- `academic_interests.py`: Script to analyze and visualize students' academic interests.
- `ClubMembership.py`: Script to analyze students' club memberships and participation.
- `extracurricular_activities.py`: Script to analyze students' extracurricular activities.
- `main.py`: The main script to run the analysis.
- `major.py`: Script to analyze students' major selections.
- `research.py`: Script to analyze students' research projects.
- `skills.py`: Script to analyze and visualize students' skills.
