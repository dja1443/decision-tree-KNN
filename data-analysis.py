# I will be building a project for decision tree and KNN for the ait quality data set from UCI




import pandas as pd
import matplotlib.pyplot as plt
import sns

'''Step 1 & 2: data Analysis'''


file_path = "AirQualityUCI.csv"
air_quality_df = pd.read_csv(file_path, sep=';', decimal=',', header=0)
data = pd.read_csv(file_path)

air_quality_df.replace(-200, pd.NA, inplace=True)
air_quality_df = air_quality_df.loc[:, ~air_quality_df.columns.str.contains('^Unnamed')]

total_missing_values = air_quality_df.isnull().sum().sum()


print("\nFirst Few Rows of the Dataset:")
print(air_quality_df.head())

variables_data = {
    "Variable Name": [
        "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
        "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
    ],
    "Role": [
        "Feature", "Feature", "Feature", "Feature", "Feature", "Feature",
        "Feature", "Feature", "Feature", "Feature",
        "Feature", "Feature", "Feature", "Feature", "Feature"
    ],
    "Type": [
        "Date", "Categorical", "Integer", "Categorical", "Integer", "Continuous",
        "Categorical", "Integer", "Categorical", "Integer",
        "Categorical", "Categorical", "Continuous", "Continuous", "Continuous"
    ],
    "Description": [
        "Date of the measurement", "Time of the measurement",
        "True hourly averaged concentration CO in mg/m^3 (reference analyzer)",
        "Hourly averaged sensor response (nominally CO targeted)",
        "True hourly averaged overall Non-Methanic Hydrocarbons concentration in microg/m^3",
        "True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)",
        "Hourly averaged sensor response (nominally NMHC targeted)",
        "True hourly averaged NOx concentration in ppb (reference analyzer)",
        "Hourly averaged sensor response (nominally NOx targeted)",
        "True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)",
        "Hourly averaged sensor response (nominally NO2 targeted)",
        "Hourly averaged sensor response (nominally O3 targeted)",
        "Temperature in Celsius", "Relative Humidity in %", "Absolute Humidity"
    ],
    "Units": [
        "", "", "mg/m^3", "", "microg/m^3", "microg/m^3",
        "", "ppb", "", "microg/m^3",
        "", "", "Celsius", "%", "g/m^3"
    ]
}

variables_table = pd.DataFrame(variables_data)
print("\nVariables Table:")
print(variables_table.to_markdown(index=False))


for col in air_quality_df.columns:
    air_quality_df[col] = pd.to_numeric(air_quality_df[col], errors='coerce')

air_quality_df.columns = air_quality_df.columns.str.strip()

numerical_columns = air_quality_df.drop(columns=['Date', 'Time'], errors='ignore').select_dtypes(include=['number'])
summary_statistics = numerical_columns.describe().transpose()
numerical_columns = air_quality_df.select_dtypes(include=['number'])

summary_statistics['count'] = summary_statistics['count'].astype(int)
columns_to_round = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
summary_statistics[columns_to_round] = summary_statistics[columns_to_round].round(3)

print("\nSummary Statistics for Numerical Features:")
print(summary_statistics)

missing_values = air_quality_df.isnull().sum()
missing_values_excluded = missing_values.drop(['Date', 'Time'], errors='ignore')

print("\nMissing Values (Count and Proportion):")
missing_proportion = (missing_values_excluded / air_quality_df.shape[0]) * 100
missing_proportion = missing_proportion.round(2)
missing_summary = pd.DataFrame({
    "Missing Count": missing_values_excluded,
    "Missing Proportion (%)": missing_proportion
})
print(missing_summary, end="\n\n")
print(f"Total Missing Values in the Dataset: {total_missing_values}")

def detect_outliers_iqr(df, columns):
    outlier_summary = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_summary[column] = len(outliers)
    return outlier_summary

outlier_stats = detect_outliers_iqr(air_quality_df, numerical_columns)
print("\nOutliers Detected Per Column:")
for column, count in outlier_stats.items():
    print(f"{column}: {count}")

file_path = "Cleaned_AirQualityUCI.csv"
air_quality_df_imputed = pd.read_csv(file_path)
numerical_columns = air_quality_df_imputed.drop(columns=['Date', 'Time'], errors='ignore')
numerical_columns = numerical_columns.loc[:, ~numerical_columns.columns.str.contains('^Unnamed')]
summary_stats = numerical_columns.describe().transpose()
columns_to_round = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
summary_stats['count'] = summary_stats['count'].astype(int)
summary_stats[columns_to_round] = summary_stats[columns_to_round].round(1)
print("\nSummary Statistics for Numerical Features after Imputation:")
print(summary_stats)


numerical_columns.hist(figsize=(10, 8), bins=20, edgecolor='k')
plt.suptitle('Distribution of Numerical Features')
plt.show()



