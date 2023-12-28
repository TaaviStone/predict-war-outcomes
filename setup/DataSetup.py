import pandas as pd


def output_missing_data(df, name):
    missing_data = df.isnull().sum()
    print(f"Missing values in {name}: {missing_data}")


def load_data(file_path):
    return pd.read_csv(file_path)


def merge_dataframes(dfs, on_column):
    merged_df = pd.concat(dfs, axis=1, join='inner')
    return merged_df


def fill_missing_values(df, column, value):
    df[column] = df[column].fillna(value)
    return df


def drop_columns_and_missing_values(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop).dropna()
    return df


def filter_draws(df, column):
    df = df[df[column] != 0]
    return df


def create_dummies_and_drop_column(df, column, prefix):
    df[column] = df[column].map(lambda ca: ca[0])
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, inplace=True, axis=1)
    return df


def preprocess_weather_column(df, column, value_map):
    df[column] = df[column].map(value_map)
    return df


def preprocess_terrain_column(df, column, value_map):
    df[column] = df[column].map(value_map)
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, inplace=True, axis=1)
    return df


def preprocess_combined_data(df):
    df['wina'] = df['wina'].apply(lambda x: x + 1 if x == -1 else x)
    df = df.astype('int')
    return df


def main():
    # Load data
    battles = load_data('../battle_files/battles.csv')
    terrain = load_data('../battle_files/terrain.csv')
    weather = load_data('../battle_files/weather.csv')
    front_width = load_data('../battle_files/front_widths.csv')

    # Merge dataframes
    dfs_to_merge = [battles, terrain, weather, front_width]
    merged_df = merge_dataframes(dfs_to_merge, 'isqno')

    # Handle missing values
    merged_df = fill_missing_values(merged_df, 'wina', -1)
    output_missing_data(merged_df[['surpa', 'wina']], 'surprise')
    output_missing_data(merged_df[['terra1', 'terra2', 'terra3', 'wina']], 'terrain')
    output_missing_data(merged_df[['wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'wina']], 'weather')

    # Drop columns and handle missing values
    columns_to_drop = ['wx2', 'terra3']
    combined_df = drop_columns_and_missing_values(merged_df, columns_to_drop)

    # Filter draws
    combined_df = filter_draws(combined_df, 'wina')

    # Preprocess columns
    combined_df = create_dummies_and_drop_column(combined_df, 'post1', 'post1')
    combined_df = preprocess_weather_column(combined_df, 'wx1', {'W': 1, 'D': 0})
    combined_df = create_dummies_and_drop_column(combined_df, 'wx3', 'wx3')
    combined_df = create_dummies_and_drop_column(combined_df, 'wx4', 'wx4')
    combined_df = create_dummies_and_drop_column(combined_df, 'wx5', 'wx5')
    combined_df = preprocess_terrain_column(combined_df, 'terra1', {'R': 0, 'G': 1, 'F': 0})
    combined_df = preprocess_terrain_column(combined_df, 'terra2', {'B': 0, 'M': 1, 'D': 0, 'W': 1})

    # Save preprocessed data
    combined_df.to_csv('../Models/combined_data.csv', index=False)


if __name__ == "__main__":
    main()
