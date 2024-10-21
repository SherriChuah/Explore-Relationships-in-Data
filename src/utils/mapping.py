import calendar
from utils.mapping_constant import (
    mapping_places_profile5,
    mapping_places_profile4,
    mapping_places_profile2,
    mapping_places_profile7,
    mapping_places_profile1,
    mapping_places_profile3,
    mapping_places_profile9,
    mapping_places_profile12,
    mapping_places_profile6,
    mapping_places_profile11,
    mapping_places_profile10
)


def map_time(time):
    hour = time.hour
    if 5 <= hour < 11:
        return 'breakfast'
    elif 11 <= hour < 14:
        return 'lunch'
    elif 14 <= hour < 18:
        return 'snack'
    elif 18 <= hour < 21:
        return 'dinner'
    else:
        return 'after dinner'


def map_time_encoded(time):
    hour = time.hour

    if 5 <= hour < 11:
        return 0
    elif 11 <= hour < 14:
        return 1
    elif 14 <= hour < 18:
        return 2
    elif 18 <= hour < 21:
        return 3
    else:
        return 4


def map_date(date):
    return calendar.day_name[date.weekday()]


def map_date_encoded(date):
    return date.weekday()


###################################
#          Common function
###################################
def profile_bank_statement_mapping(df_profile_analysis, mapping_number):
    df_profile_analysis['Time_category'] = df_profile_analysis['Time'].apply(
        lambda x: map_time(x))

    df_profile_analysis['Time_category_encoded'] = df_profile_analysis[
        'Time'].apply(lambda x: map_time_encoded(x))

    df_profile_analysis['Date_category'] = df_profile_analysis['Date'].apply(
        lambda x: map_date(x))

    df_profile_analysis['Date_category_encoded'] = df_profile_analysis[
        'Date'].apply(lambda x: map_date_encoded(x))

    if mapping_number == 4:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis["Name"].apply(
            lambda x: mapping_places_profile4[x]
            if x in mapping_places_profile4 else None
        )
    elif mapping_number == 5:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile5[x]
        )

    elif mapping_number == 2:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile2[x]
            if x in mapping_places_profile2 else None
        )

    elif mapping_number == 7:
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Description"].apply(
            lambda x: mapping_places_profile7[x]
            if x in mapping_places_profile7 else None
        )

    elif mapping_number == 1:
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Description"].apply(
            lambda x: mapping_places_profile1[x]
            if x in mapping_places_profile1 else None
        )

    elif mapping_number == 3:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile3[x]
            if x in mapping_places_profile3 else None
        )

    elif mapping_number == 9:
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile9[x]
            if x in mapping_places_profile9 else None
        )

    elif mapping_number == 12:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile12[x]
            if x in mapping_places_profile12 else None
        )

    elif mapping_number == 6:
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Transaction Description"].apply(
            lambda x: mapping_places_profile6[x]
            if x in mapping_places_profile6 else None
        )

    elif mapping_number == 11:
        df_profile_analysis = df_profile_analysis.drop(columns=['Description'])
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Name"].apply(
            lambda x: mapping_places_profile11[x]
            if x in mapping_places_profile11 else None
        )

    elif mapping_number == 10:
        df_profile_analysis["Name_category"] = df_profile_analysis[
            "Description"].apply(
            lambda x: mapping_places_profile10[x]
            if x in mapping_places_profile10 else None
        )

    df_profile_analysis = df_profile_analysis[df_profile_analysis['Name_category'].notna()]

    return df_profile_analysis


def profile_food_diary_mapping(df):
    df['Time_category'] = df['Time'].apply(lambda x: map_time(x))

    df['Time_category_encoded'] = df['Time'].apply(
        lambda x: map_time_encoded(x))

    df['Date_category'] = df['Date'].apply(lambda x: map_date(x))

    df['Date_category_encoded'] = df['Date'].apply(
        lambda x: map_date_encoded(x))

    return df
