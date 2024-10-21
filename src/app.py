import streamlit as st

from streamlit_helper.streamlit_helper import display_html_graph, \
    get_list_of_participants, get_graphs, display_overview_graph, \
    display_dbscan_clustering_analysis, display_k_means_analysis, \
    display_comparing_bank_statement_and_food_diary, \
    display_k_means_analysis_bank_statement

# Page Config of app
st.set_page_config(
    layout="wide"  # Options are "centered" or "wide"
)

# Title of app
st.title("Investigate relationships between one's bank statements and their food profile")


# Create two tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Person Analysis", "Predictions"])


with tab1:
    st.header("Bank Statement and Food Diary Overview")
    # Dropdown for selecting which overview
    view_selected = st.selectbox(
        "Select a view:",
        ['Bank Statement - Day Log Frequency Per Person',
         'Bank Statement - Meal Type Count Per Person',
         'Bank Statement - Food Type Count Per Person',
         'Food Diary - Food Log Count Per Person'
         ]
    )

    view_map = {
        'Bank Statement - Day Log Frequency Per Person': 'Number_of_days_logged_per_person.json',
        'Bank Statement - Meal Type Count Per Person': 'Meal_type_count_per_person.json',
        'Bank Statement - Food Type Count Per Person': 'Food_type_count_per_person.json',
        'Food Diary - Food Log Count Per Person': 'Food_log_count_per_person.json'
    }
    display_overview_graph(view_map[view_selected])

with tab2:
    list_of_participants = get_list_of_participants()

    # Plot the selected graph
    st.header("Person Bank Statement Analysis")

    # Dropdown for selecting the persons
    participant_selected = st.selectbox(
        "Select a Participant:", list_of_participants
        )

    participants_graphs = {
        'Profile 1': get_graphs('Person 1'),
        'Profile 2': get_graphs('Person 2'),
        'Profile 3': get_graphs('Person 3'),
        'Profile 4': get_graphs('Person 4'),
        'Profile 5': get_graphs('Person 5'),
        'Profile 6': get_graphs('Person 6'),
        'Profile 7': get_graphs('Person 7'),
        'Profile 8': None,
        'Profile 9': get_graphs('Person 9'),
        'Profile 10': get_graphs('Person 10'),
        'Profile 11': get_graphs('Person 11'),
        'Profile 12': get_graphs('Person 12'),
    }

    graphs_dic = participants_graphs[participant_selected]

    display_html_graph(graphs_dic)

with tab3:
    st.header("Using DBSCAN and K-means to cluster Food in Food Diary and Bank Statements.")

    tab1_1, tab2_1, tab3_1, tab4_1 = st.tabs(
        ["DBSCAN to cluster Food Diary",
         "K means to cluster Food Diary",
         "K means to cluster Bank Statements",
         "Comparing Bank Statements and Food Diary"])

    with tab1_1:
        display_dbscan_clustering_analysis()

    with tab2_1:
        display_k_means_analysis()

    with tab3_1:
        display_k_means_analysis_bank_statement()

    with tab4_1:
        display_comparing_bank_statement_and_food_diary()
