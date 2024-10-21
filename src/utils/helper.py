import sweetviz as sv
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.io as pio
from utils.mapping import profile_food_diary_mapping
import os

def sweetviz_visualise(df):
    report = sv.analyze(df)
    report.show_notebook()


def chi_squared_test_relationship(df, col1, col2, alpha):
    contingency_table = pd.crosstab(
        df[col1],
        df[col2],
    )

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f'Chi-Squared Statistic: {chi2}, P-value: {p}')

    # Decision
    if p < alpha:
        print(
            "Reject the null hypothesis: There is a relationship between the two categories.")
    else:
        print(
            "Fail to reject the null hypothesis: No significant relationship.")

    return chi2, contingency_table


def cramers_v_get_strength(df, col1, col2):
    contingency_table = pd.crosstab(
        df[col1],
        df[col2],
    )

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    k = contingency_table.shape[0]
    r = contingency_table.shape[1]

    cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))

    print(f"Cramér's V: {cramers_v}")


def cramers_v(chi2, n, cat1, cat2):
    return np.sqrt(chi2 / (n * (min(cat1, cat2) - 1)))


def generate_graph(df, x, y, color, title, labels, order):

    fig = px.bar(df, x=x, y=y, color=color, title=title, labels=labels,
                 category_orders={'Time_category': order})

    # Update layout for better readability
    fig.update_layout(
        barmode="stack",
        legend={'traceorder': 'normal'}
    )

    if x == "Date_category":
        fig.update_xaxes(categoryorder='array',
                         categoryarray=[
                             "Monday", "Tuesday", "Wednesday",
                             "Thursday", "Friday", "Saturday", "Sunday"
                         ])
    elif x == "Time_category":
        fig.update_xaxes(categoryorder='array',
                         categoryarray=[
                             "breakfast", 'lunch', 'snack', 'dinner', 'after dinner'
                         ])

    # Show the plot
    fig.show()

    return fig


def get_time_and_date_splitting(data):
    day = data.split(' ')[0]
    time = data.split(' ')[1]

    return pd.Series([day, time])


def generate_k_means_and_pca(df):
    df_for_clustering = df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_for_clustering)

    # PCA
    X_pca = generate_pca_graph(df_for_clustering, X_scaled)

    # K means
    df_cluster = generate_k_means(df, X_pca)

    # Graph cluster and count
    df_x_pca = pd.DataFrame(X_pca)
    fig = px.scatter(
        df_x_pca, 0, 1,
        color=df_for_clustering['Cluster'],
        title='K means Clustering after PCA'
    )

    fig.update_traces(marker={'size': 15})

    fig.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
    )
    fig.show()

    return X_pca, df_cluster, fig


def generate_dbscan_and_pca(df, min_samples=5):
    df_for_clustering = df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_for_clustering)

    # PCA
    X_pca = generate_pca_graph(df_for_clustering, X_scaled)

    # DBSCAN
    eps = 0.5
    min_samples = min_samples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_for_clustering['Cluster'] = dbscan.fit_predict(X_pca)

    # Graph cluster and count
    df_x_pca = pd.DataFrame(X_pca)
    fig = px.scatter(
        df_x_pca,
        0, 1,
        color=df_for_clustering['Cluster'],
        title='DBSCAN Clustering after PCA'
    )

    fig.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Let's us know which food entry in the diary are clustered together",
                xref='paper', yref='paper',
                x=0, y=1.1,  # Position above the title
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    fig.show()

    return X_pca, df_for_clustering, fig


def generate_k_means(df, X_scaled):
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

    optimal_k = int(input("Cluster Count: "))

    # Apply K-Means with the chosen number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # # Get cluster centers and scale them back to original
    # centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # cluster_centers_df = pd.DataFrame(centers, columns=df.columns[:len(
    #     df.columns) - 1])

    return df


def generate_pca_graph(df, X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
    # plt.title('K-Means Clustering of Meals')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.colorbar(label='Cluster')
    # plt.show()

    return X_pca


def fig_to_dataframe_1(fig, x_name, y_name):
    data = []
    for trace in fig.data:
        for x, y in zip(trace.x, trace.y):
            data.append({
                x_name: x,
                y_name: y,
            })
    return pd.DataFrame(data)


def get_day_log_count_per_participant(date_distribution_path):
    person1_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[0]), "Day", "Count"
    )
    person1_day_log_count["Person"] = "1"

    person2_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[1]), "Day", "Count"
    )
    person2_day_log_count["Person"] = "2"

    person3_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[2]), "Day", "Count"
    )
    person3_day_log_count["Person"] = "3"

    person4_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[3]), "Day", "Count"
    )
    person4_day_log_count["Person"] = "4"

    person5_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[4]), "Day", "Count"
    )
    person5_day_log_count["Person"] = "5"

    person6_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[5]), "Day", "Count"
    )
    person6_day_log_count["Person"] = "6"

    person7_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[6]), "Day", "Count"
    )
    person7_day_log_count["Person"] = "7"

    person9_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[8]), "Day", "Count"
    )
    person9_day_log_count["Person"] = "9"

    person10_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[9]), "Day", "Count"
    )
    person10_day_log_count["Person"] = "10"
    print(person10_day_log_count)

    person11_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[10]), "Day", "Count"
    )
    person11_day_log_count["Person"] = "11"

    person12_day_log_count = fig_to_dataframe_1(
        pio.read_json(date_distribution_path[11]), "Day", "Count"
    )
    person12_day_log_count["Person"] = "12"

    return [
        person1_day_log_count,
        person2_day_log_count,
        person3_day_log_count,
        person4_day_log_count,
        person5_day_log_count,
        person6_day_log_count,
        person7_day_log_count,
        person9_day_log_count,
        person10_day_log_count,
        person11_day_log_count,
        person12_day_log_count
    ]


def fig_to_dataframe(fig, x_name, y_name, name_name):
    data = []

    for trace in fig.data:
        for x, y in zip(trace.x, trace.y):
            data.append({
                x_name: x,
                y_name: y,
                name_name: trace.name
            })

    return pd.DataFrame(data)


def get_meal_type_count_per_participant(meal_per_day_log_distribution_path):
    person1_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[0]), "Day", "Count",
        "Meal Type"
    )
    person1_meal_type_per_day["Person"] = "1"

    person2_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[1]), "Day", "Count",
        "Meal Type"
    )
    person2_meal_type_per_day["Person"] = "2"

    person3_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[2]), "Day", "Count",
        "Meal Type"
    )
    person3_meal_type_per_day["Person"] = "3"

    person4_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[3]), "Day", "Count",
        "Meal Type"
    )
    person4_meal_type_per_day["Person"] = "4"

    person5_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[4]), "Day", "Count",
        "Meal Type"
    )
    person5_meal_type_per_day["Person"] = "5"

    person6_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[5]), "Day", "Count",
        "Meal Type"
    )
    person6_meal_type_per_day["Person"] = "6"

    person7_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[6]), "Day", "Count",
        "Meal Type"
    )
    person7_meal_type_per_day["Person"] = "7"

    person9_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[8]), "Day", "Count",
        "Meal Type"
    )
    person9_meal_type_per_day["Person"] = "9"

    person10_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[9]), "Day", "Count",
        "Meal Type"
    )
    person10_meal_type_per_day["Person"] = "10"

    person11_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[10]), "Day", "Count",
        "Meal Type"
    )
    person11_meal_type_per_day["Person"] = "11"

    person12_meal_type_per_day = fig_to_dataframe(
        pio.read_json(meal_per_day_log_distribution_path[11]), "Day", "Count",
        "Meal Type"
    )
    person12_meal_type_per_day["Person"] = "12"

    return [
        person1_meal_type_per_day,
        person2_meal_type_per_day,
        person3_meal_type_per_day,
        person4_meal_type_per_day,
        person5_meal_type_per_day,
        person6_meal_type_per_day,
        person7_meal_type_per_day,
        person9_meal_type_per_day,
        person10_meal_type_per_day,
        person11_meal_type_per_day,
        person12_meal_type_per_day
    ]


def standardise_counts(x, y, mapping_day_log_count):
    value = y / mapping_day_log_count[str(x)]
    return value


def get_food_type_count_per_participant(food_type_per_day_count_distribution_path):
    person1_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[0]),
        "Day",
        "Count",
        "Food Type",
    )
    person1_food_type_per_person["Person"] = "1"

    person2_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[1]),
        "Day",
        "Count",
        "Food Type",
    )
    person2_food_type_per_person["Person"] = "2"

    person3_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[2]),
        "Day",
        "Count",
        "Food Type",
    )
    person3_food_type_per_person["Person"] = "3"

    person4_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[3]),
        "Day",
        "Count",
        "Food Type",
    )
    person4_food_type_per_person["Person"] = "4"

    person5_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[4]),
        "Day",
        "Count",
        "Food Type",
    )
    person5_food_type_per_person["Person"] = "5"

    person6_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[5]),
        "Day",
        "Count",
        "Food Type",
    )
    person6_food_type_per_person["Person"] = "6"

    person7_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[6]),
        "Day",
        "Count",
        "Food Type",
    )
    person7_food_type_per_person["Person"] = "7"

    person9_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[8]),
        "Day",
        "Count",
        "Food Type",
    )
    person9_food_type_per_person["Person"] = "9"

    person10_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[9]),
        "Day",
        "Count",
        "Food Type",
    )
    person10_food_type_per_person["Person"] = "10"

    person11_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[10]),
        "Day",
        "Count",
        "Food Type",
    )
    person11_food_type_per_person["Person"] = "11"

    person12_food_type_per_person = fig_to_dataframe(
        pio.read_json(food_type_per_day_count_distribution_path[11]),
        "Day",
        "Count",
        "Food Type",
    )
    person12_food_type_per_person["Person"] = "12"

    return [
        person1_food_type_per_person,
        person2_food_type_per_person,
        person3_food_type_per_person,
        person4_food_type_per_person,
        person5_food_type_per_person,
        person6_food_type_per_person,
        person7_food_type_per_person,
        person9_food_type_per_person,
        person10_food_type_per_person,
        person11_food_type_per_person,
        person12_food_type_per_person
    ]


def prepare_food_diary_for_classification(df_food_diary):
    df_food_diary = df_food_diary[df_food_diary['Participant'].notna()]
    df_food_diary = df_food_diary[
        df_food_diary['Participant'].str.startswith('Profile')]
    df_food_diary['Date'] = pd.to_datetime(df_food_diary['Date'],
                                           format='%m/%d/%Y')
    df_food_diary['Time'] = pd.to_datetime(df_food_diary['Time'],
                                           format='%H:%M')

    df_food_diary = df_food_diary[
        [
            "Participant",
            "Date",
            "Time",
            "Kcal",
            "Carbs (g)",
            "Proteins (g)",
            "Sugars",
            "Fats",
            "Saturated Fats",
            "Salt",
            "Fibre",
            "Micronutrients"
        ]
    ]

    df_food_diary = profile_food_diary_mapping(df_food_diary)

    df_food = df_food_diary[[
        "Kcal",
        "Carbs (g)",
        "Proteins (g)",
        "Sugars",
        "Fats",
        "Saturated Fats",
        "Salt",
        "Fibre",
        "Micronutrients",
        "Time_category_encoded",
        "Date_category_encoded",
    ]]

    return df_food_diary, df_food
