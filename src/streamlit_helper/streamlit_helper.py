import streamlit as st
import os
import pandas as pd
import plotly.io as pio
from collections import defaultdict

from utils.helper import prepare_food_diary_for_classification

curr_path = os.getcwd()
curr_dir = os.path.dirname(curr_path)


def get_list_of_participants():
    food_diary_path = os.path.join(
        curr_dir,
        "src/Data/Food Diary/Health project.csv",
    )
    df_food_diary = pd.read_csv(food_diary_path)

    df_food_diary, df_food = prepare_food_diary_for_classification(df_food_diary)

    unsorted_list = list(set(df_food_diary['Participant']))

    return sorted(unsorted_list, key=lambda x: int(x.split()[1]))


# Function to display an HTML graph
def display_html_graph(graph_dict):
    # Create two rows
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    count = 1
    for graph_name, graph in graph_dict.items():
        if count == 1:
            with row1_col1:
                st.plotly_chart(graph)
        elif count == 2:
            with row1_col2:
                st.plotly_chart(graph)
        elif count == 3:
            with row2_col1:
                st.plotly_chart(graph)
        elif count == 4:
            with row2_col2:
                st.plotly_chart(graph)
        count += 1


def get_graphs(person):
    person_figures_folder_path = os.path.join(
        curr_dir,
        f"src/figures/{person}",
    )
    html_graphs = defaultdict()

    # List all files in the given folder
    for file_name in os.listdir(person_figures_folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(person_figures_folder_path, file_name)
            fig = pio.read_json(file_path)
            html_graphs[file_name] = fig

    return html_graphs


def display_overview_graph(view_selected):
    st.subheader("Observations / Assumptions")
    if view_selected == 'Number_of_days_logged_per_person.json':
        st.write("- In 2-3 months, there are around 8-13 days separately for each day.")
        st.write("- Person 8's bank statement is in PDF, unable to extract.")
        st.write("- Person 4, 10 and 12 does not have many entries in their bank statements.")
    elif view_selected == 'Meal_type_count_per_person.json':
        meal_type = {
            "Breakfast": ['5 am to 11 am'],
            "Lunch": ['11 am to 2 pm'],
            "Snack": ["2 pm to 6 pm"],
            "Dinner": ["6 pm to 9 pm"],
            "After Dinner": ["9 pm to 5 am"]
        }
        df = pd.DataFrame(meal_type)
        st.write(
            "- Count is standardise by dividing each by the total number of meals in bank statement respectively to make better comparisons.")
        st.write("- Meal type category split by: ")
        st.write(df)
        st.write("- Meal time does not represent the exact moment a person eats thus meal type may not be accurate.")
    elif view_selected == 'Food_type_count_per_person.json':
        st.write(
            "- Count is standardise by dividing each by the total number of meals in bank statement respectively to make better comparisons.")
        st.write("- Food type category of bank statement transactions are manually categorised by me. Could streamline by using external api to get a better category.")
        st.write("- Knowing food type in the restaurant visited would help understand patterns.")
    else:
        st.write("- Some meals were missed, might affect outcome when clustering.")
        st.write("- Drinks were not taken which will affect outcome as well.")

    path = os.path.join(
        curr_dir,
        f"src/figures/Consolidated/{view_selected}",
    )
    fig = pio.read_json(path)
    st.plotly_chart(fig, key=1)


def display_dbscan_clustering_analysis():
    path = os.path.join(curr_dir,
                        "src/figures/Consolidated/dbcsan_cluster.json")
    fig_dbscan = pio.read_json(path)

    path = os.path.join(curr_dir,
                        "src/figures/Consolidated/Mean_feature_per_cluster.json")
    fig_mean_feature_per_cluster = pio.read_json(path)

    path = os.path.join(curr_dir,
                        "src/figures/Consolidated/Food_diary_cluster_count_per_person.json")
    fig_food_cluster_count = pio.read_json(path)

    graph_dict = {
        "dbscan": fig_dbscan,
        "food_cluster_count": fig_food_cluster_count,
    }

    st.subheader("Observations / Assumptions")
    st.write("- Food Diary entries were clustered into 2 clusters, 0 and -1")
    st.write("- Cluster 0 might represent food that is low in most nutrients")
    st.write("- Cluster 1 might represent food that is high in most nutrients")

    display_html_graph(graph_dict)

    st.plotly_chart(fig_mean_feature_per_cluster)


def display_k_means_analysis():
    path = os.path.join(curr_dir,
                        "src/figures/Consolidated/Classify_person_food_diary.json")
    fig_kmeans = pio.read_json(path)

    st.subheader("Observations / Assumptions")
    st.write("- Summed up all the nutrients associated to the Person and cluster it using K means.")
    st.write("- 3 clusters appear from the model.")

    cluster_result = {
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Persons": [
            'Profile 10, Profile 11, Profile 2, Profile 5, Profile 6, Profile 7, Profile 9',
            "Profile 8",
            'Profile 1, Profile 12, Profile 3, Profile 4']
    }
    st.write(pd.DataFrame(cluster_result))
    st.plotly_chart(fig_kmeans)


def display_k_means_analysis_bank_statement():
    path = os.path.join(curr_dir,
                        "src/figures/Consolidated/k_means_cluster_bank_statement.json")
    fig_kmeans = pio.read_json(path)

    st.subheader("Observations / Assumptions")
    st.write("- Cluster bank statements based on frequency of Food Type visits.")
    st.write("- 3 clusters appear from the model.")

    cluster_result = {
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Persons": [
            'Profile 1, Profile 2, Profile 3, Profile 5, Profile 9',
            "Profile 4, Profile 7, Profile 11",
            'Profile 6, Profile 10, Profile 12']
    }
    st.write(pd.DataFrame(cluster_result))
    st.plotly_chart(fig_kmeans)

def display_comparing_bank_statement_and_food_diary():
    st.subheader("Observations / Improvements")
    st.write("- Need more consistent information and longer duration to infer relationship.")
    st.write("- Breakdown of what type of food was consumed will help with accuracy.")
    st.write("- Difficult to breakdown what was bought in supermarkets.")
    st.write("- Might require more participants.")

    st.write("Cluster Food Diary Result")
    cluster_result = {
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Cluster Interpretation": ['Unsure, maybe category 0', 'Unhealthy, category 2', 'Unsure, maybe category 1'],
        "Persons": [
            'Profile 2, Profile 5, Profile 6, Profile 7, Profile 9, Profile 10, Profile 11',
            "Profile 8",
            'Profile 1, Profile 3, Profile 4, Profile 12']
    }
    st.write(pd.DataFrame(cluster_result))

    st.write("Cluster Bank Statement Result")
    cluster_result = {
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Cluster Interpretation": ['Unsure, maybe category 1',
                                   'Unsure, maybe category 0',
                                   'Unsure'],
        "Persons": [
            'Profile 1, Profile 2, Profile 3, Profile 5, Profile 9',
            "Profile 4, Profile 7, Profile 11",
            'Profile 6, Profile 10, Profile 12']
    }
    st.write(pd.DataFrame(cluster_result))

    st.write("Interview Result")
    cluster_result = {
        "category": ["Healthy - Vege, Vegan, IF, Low Card", "Regular Eating", "Regular Consumption of Large Quantity of Meat and Ultra Processed Food"],
        "Persons": [
            'Profile 4, Profile 7, Profile 10, Profile 11',
            'Profile 1, Profile 3, Profile 5, Profile 9, Profile 12',
            'Profile 2, Profile 6, Profile 8']
    }
    st.write(pd.DataFrame(cluster_result))

    path = os.path.join(
        curr_dir,
        f"src/figures/Consolidated/Food_type_count_per_person.json",
    )
    fig = pio.read_json(path)
    fig.update_layout(height=400)
    st.plotly_chart(fig)
