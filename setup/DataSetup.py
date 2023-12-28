import pandas as pd


def output_missing_data(df, label):
    missing_data = df.isnull().sum()
    print(f"Missing values {label}: ", missing_data)


def process_categorical_feature(df, column, prefix=None):
    df[column] = df[column].map(lambda value: value[0])
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, inplace=True, axis=1)
    return df


def process_binary_feature(df, column, true_value='W'):
    df[column] = df[column].map(lambda value: 1 if value == true_value else 0)
    return df


# Read data from CSV files
battles = pd.read_csv('../battle_files/battles.csv')
terrain = pd.read_csv('../battle_files/terrain.csv')
weather = pd.read_csv('../battle_files/weather.csv')
front_width = pd.read_csv('../battle_files/front_widths.csv')

# Select only WW2 battles
war_list = ['WORLD WAR II (ITALY 1943-1944)', 'WORLD WAR II (ITALY 1944)', 'WORLD WAR II (EUROPEAN THEATER)',
            'WORLD WAR II', 'WORLD WAR II (EASTERN FRONT)', 'WORLD WAR II (OKINAWA)']

# Merge relevant tables
merged_df = pd.merge(battles, terrain, on="isqno")
merged_df = pd.merge(merged_df, weather, on="isqno")
merged_df = pd.merge(merged_df, front_width, on="isqno")
merged_df.set_index('isqno', inplace=True)
merged_df = merged_df[
    ['surpa', 'post1', 'post2', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'terra3', 'aeroa', 'wina',
     'wofa', 'wofd']]

# Fill NaN values in 'wina' with -1, indicating that the attacker lost
merged_df['wina'] = merged_df['wina'].fillna(-1)

# Check and output missing data for different features
output_missing_data(merged_df[['surpa', 'wina']], 'Surprise')
output_missing_data(merged_df[['terra1', 'terra2', 'terra3', 'wina']], 'Terrain')
output_missing_data(merged_df[['wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'wina']], 'Weather')
output_missing_data(merged_df[['post1', 'post2', 'wina']], 'Fortification')
output_missing_data(merged_df[['aeroa', 'wina']], 'Aerial Super')
output_missing_data(merged_df[['wofa', 'wofd', 'wina']], 'Frontline Width')

# Save cleaned data to CSV files
merged_df[['surpa', 'wina']].to_csv('../Graphs/surprise_data.csv', index=False)
merged_df[['terra1', 'terra2', 'terra3', 'wina']].to_csv('../Graphs/terrain_data.csv', index=False)
merged_df[['wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'wina']].to_csv('../Graphs/weather_data.csv', index=False)
merged_df[['aeroa', 'wina']].to_csv('../Graphs/aerialSuper_data.csv', index=False)

# Exclude features with too many missing values for imputation (post2, terra3)
combined_df = merged_df[
    ['surpa', 'post1', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'aeroa', 'wofa', 'wofd', 'wina']]
print("Missing values: ", combined_df.isnull().sum())

# Drop 'wx2' column due to too many missing values and remove remaining NaNs
combined_df = combined_df.drop('wx2', axis=1).dropna()
print("Missing values after dropping: ", combined_df.isnull().sum())

# Exclude draws from the dataset
combined_df = combined_df[combined_df['wina'] != 0]

# Process categorical and binary features
combined_df = process_categorical_feature(combined_df, 'post1', 'post1')
combined_df = process_binary_feature(combined_df, 'wx1')
combined_df = process_categorical_feature(combined_df, 'wx3', 'wx3')
combined_df = process_categorical_feature(combined_df, 'wx4', 'wx4')
combined_df = process_categorical_feature(combined_df, 'wx5', 'wx5')
combined_df = process_categorical_feature(combined_df, 'terra1', 'terra1')
combined_df = process_categorical_feature(combined_df, 'terra2', 'terra2')

# Prepare data for modeling
combined_df['wina'] = combined_df['wina'].apply(lambda x: x + 1 if x == -1 else x)
combined_df = combined_df.astype('int')

# Save the final cleaned and preprocessed data to a CSV file
combined_df.to_csv('../Models/combined_data.csv', index=False)
