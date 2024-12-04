import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer


file_path = "AirQualityUCI.csv"
air_quality_df = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)


air_quality_df = air_quality_df.loc[:, ~air_quality_df.columns.str.contains('^Unnamed')]

air_quality_df.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')

threshold = 0.5 * len(air_quality_df)

columns_to_drop = air_quality_df.columns[air_quality_df.isnull().sum() > threshold]
air_quality_df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns with >50% missing values: {list(columns_to_drop)}")


numerical_columns = air_quality_df.select_dtypes(include=['number']).columns
air_quality_df[numerical_columns] = air_quality_df[numerical_columns].fillna(air_quality_df[numerical_columns].mean().round(1))


cleaned_file_path = "Cleaned_AirQualityUCI.csv"
air_quality_df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")


print("\nSummary of the Cleaned Dataset:")
print(air_quality_df.info())

cleaned_file_path = "Cleaned_AirQualityUCI.csv"
cleaned_data = pd.read_csv(cleaned_file_path)

standard_scaler = StandardScaler()
numerical_columns = cleaned_data.select_dtypes(include=['number']).columns
cleaned_data[numerical_columns] = standard_scaler.fit_transform(cleaned_data[numerical_columns]).round(1)

standardized_file_path = "Standardized_AirQualityUCI.csv"
cleaned_data.to_csv(standardized_file_path, index=False)

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
cleaned_data[numerical_columns] = discretizer.fit_transform(cleaned_data[numerical_columns])

discretized_file_path = "Discretized_AirQualityUCI.csv"
cleaned_data.to_csv(discretized_file_path, index=False)

print("\nPreview of Standardized Dataset:")
print(pd.read_csv(standardized_file_path).head())

print("\nPreview of Discretized Dataset:")
print(pd.read_csv(discretized_file_path).head())