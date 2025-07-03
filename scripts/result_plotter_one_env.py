import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Parameters
models = ["PPO", "SAC", "TD3", "DDPG"]
env = "StaticObstacleEnv-v0"  # Target environment
ave_window = 1000
feature = 'total_reward'

# Set the output file name and folder
output_folder = "figures-results"
output_filename = f"{env}-results.pdf"
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
output_path = os.path.join(output_folder, output_filename)

# Set the theme for the plot
sns.set_theme(style="darkgrid")
sns.set_context("talk")

# Initialize the plot
plt.figure(figsize=(12, 9))  # 1200x900 pixels (inches * dpi, assuming 100 dpi)
plt.gca().spines[['right', 'top']].set_visible(False)

# Set specific ticks and font size
plt.xticks([0, 1e6, 2e6], ['0', '1e6', '2e6'], fontsize=25)
plt.yticks(fontsize=25)

# Plot data for each model
for model in models:
    y_data = pd.read_csv(f'logs_backup/{env}/{env}_{model}.csv')
    plt.plot(
        y_data['timesteps'][:-(ave_window - 1)], 
        moving_average(y_data[feature], ave_window), 
        label=model
    )

# Add labels and title with adjusted font size
plt.xlabel('Timesteps', fontsize=25)
plt.ylabel('Total Reward', fontsize=25)
# plt.title(f'Reward Progression in {env}', fontsize=25)

# Adjust legend: larger font size and position in bottom-right corner
legend = plt.legend(
    fontsize=18,  # Increased font size
    loc='lower right',  # Positioned in the bottom-right corner
    frameon=True
)
legend.get_frame().set_facecolor('white')  # White background
# legend.get_frame().set_edgecolor('black')  # Optional: black border

# Save the figure to a PDF file
plt.tight_layout()
plt.savefig(output_path, format='pdf')
plt.close()  # Close the plot to free memory

print(f"Plot saved as {output_path}")