#%%
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#%%
test_ev = pd.read_pickle('/home/ps/haichao/13_Fedaral_learning/test_ev/final_data.pkl')
#%%
#
# %% Split the DataFrame into three based on conditions
total_length = len(test_ev)
# Split the DataFrame into three based on conditions
df_sz = test_ev[test_ev['car_id'].str.contains('SZ')]  # Contains 'SZ'
df_bit = test_ev[test_ev['car_id'].str.contains('BIT')]  # Contains 'BIT'
df_other = test_ev[~test_ev['car_id'].str.contains('SZ|BIT')]  # Does not contain 'SZ' or 'BIT'
# Calculate the ratio of each DataFrame
ratio_sz = len(df_sz) / total_length
ratio_bit = len(df_bit) / total_length
ratio_other = len(df_other) / total_length
# Print the ratio of each DataFrame
print(f"Ratio of DataFrame containing 'SZ': {ratio_sz:.2%}")
print(f"Ratio of DataFrame containing 'BIT': {ratio_bit:.2%}")
print(f"Ratio of DataFrame not containing 'SZ' or 'BIT': {ratio_other:.2%}")
# Count the number of rows containing 'LFPHC7PE8K1A09808'
contains_count = test_ev['car_id'].str.contains('LFPHC7PE8K1A09808', na=False).sum()
# Count the total number of rows
total_count = len(test_ev)
# Calculate the ratio
ratio = contains_count / total_count
# Print the result
print(f"Ratio of string 'LFPHC7PE8K1A09808': {ratio:.2%}")
# %%
mean_sz = df_sz['Trip Distance'].mean()
mean_bit = df_bit['Trip Distance'].mean()
mean_other = df_other['Trip Distance'].mean()
# Print the results
print(f"Mean Trip Distance in DataFrame containing 'SZ': {mean_sz:.2f}")
print(f"Mean Trip Distance in DataFrame containing 'BIT': {mean_bit:.2f}")
print(f"Mean Trip Distance in DataFrame not containing 'SZ' or 'BIT': {mean_other:.2f}")
#%%
# Define a function to calculate and print the ratio of each number in the Vehicle Model column
def print_model_ratios(df, df_name):
    if not df.empty:
        ratios = df['Vehicle Model'].value_counts(normalize=True).sort_index()
        print(f"Ratios of Vehicle Model in {df_name}:")
        for model, ratio in ratios.items():
            print(f"  Model {model}: {ratio:.2%}")
    else:
        print(f"{df_name} is empty, no Vehicle Model data.")
# Print the Vehicle Model ratios for each DataFrame
print_model_ratios(df_sz, "DataFrame containing 'SZ'")
print_model_ratios(df_bit, "DataFrame containing 'BIT'")
print_model_ratios(df_other, "DataFrame not containing 'SZ' or 'BIT'")
# %%
# Configure analysis parameters
FEATURES = [
    'Trip Distance', 
    'Accumulated Driving Range',
    'Avg. Speed',
    'Avg. of Acceleration',
    'Avg. of Deceleration',
    'Energy Recovery Ratio',
    'State of Charge',
    'Avg. Starting SOC',
    'Temperature',
    'Wind Speed'
]
GROUPS = {
    'Main client': df_other,
    'Secondary client': df_sz,
    'Minor client	': df_bit,
    
}
# Build the statistics matrix
stats_dict = {}
for group_name, df in GROUPS.items():
    group_stats = pd.DataFrame({
        'Mean': df[FEATURES].mean(numeric_only=True),
        'Std': df[FEATURES].std(ddof=0)  # Use population standard deviation
    })
    stats_dict[group_name] = group_stats
# Generate a structured report
report_df = pd.concat(stats_dict, axis=1).round(4)
report_df
# %%