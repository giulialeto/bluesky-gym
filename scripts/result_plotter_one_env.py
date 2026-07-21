import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Parameters
models = ["PPO", "SAC", "TD3", "DDPG"] # this is the order used for the BlueSky-Gym paper
models = ["SAC"]
env = "StaticObstacleCREnv-v1"  # Target environment
ave_window = 1000 
features = ['waypoint_reached', 'total_reward', 'crashed', 'average_drift']  # Features 'total_reward', 'waypoint_reached', 'crashed', 'average_drift'
features = ['total_reward', 'waypoint_reached', 'crashed', 'average_drift', 'total_intrusion_time', 'intrusion_counter', 'reach_reward', 'drift_reward', 'intrusion_other_ac_reward', 'intrusion_reward', 'average_hdg_action', 'average_spd_action', 'hdg_action_saturated_rate', 'spd_action_saturated_rate', 'average_hdg_action_delta', 'average_spd_action_delta', 'hdg_reversal_rate', 'spd_reversal_rate', 'min_cpa', 'num_encounters', 'min_obstacle_distance', 'num_obstacle_encounters', 'extra_path_length', 'path_length_ratio', 'actual_path_length', 'planned_path_length', 'planned_path_time']

labels = {
    'waypoint_reached': 'Waypoint Reached',
    'total_reward': 'Total Reward',
    'crashed': 'Crashed',
    'average_drift': 'Average Drift',
    'total_intrusion_time': 'Total Intrusion Time',
    'intrusion_counter': 'Intrusion Counter',
    'reach_reward': 'Reach Reward',
    'drift_reward': 'Drift Reward',
    'intrusion_other_ac_reward': 'Intrusion Other AC Reward',
    'intrusion_reward': 'Intrusion Reward',
    'average_hdg_action': 'Average Heading Action',
    'average_spd_action': 'Average Speed Action',
    'hdg_action_saturated_rate': 'Heading Action Saturated Rate',
    'spd_action_saturated_rate': 'Speed Action Saturated Rate',
    'average_hdg_action_delta': 'Average Heading Action Delta',
    'average_spd_action_delta': 'Average Speed Action Delta',
    'hdg_reversal_rate': 'Heading Reversal Rate',
    'spd_reversal_rate': 'Speed Reversal Rate',
    'min_cpa': 'Minimum CPA',
    'num_encounters': 'Number of Encounters',
    'min_obstacle_distance': 'Minimum Obstacle Distance',
    'num_obstacle_encounters': 'Number of Obstacle Encounters',
    'extra_path_length': 'Extra Path Length',
    'path_length_ratio': 'Path Length Ratio',
    'actual_path_length': 'Actual Path Length',
    'planned_path_length': 'Planned Path Length',
    'planned_path_time': 'Planned Path Time'
}

# Set the output file name and folder
output_folder = "figures_results"
all_models = "_".join(models)

for feature in features:
    output_filename = f"{env}_{all_models}-results_{feature}.pdf"
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
        y_data = pd.read_csv(f'scripts/common/results/logs_backup/{env}/{env}_{model}.csv')
        plt.plot(
            y_data['timesteps'][:-(ave_window - 1)], 
            moving_average(y_data[feature], ave_window), 
            label=model
        )

    # Add labels and title with adjusted font size
    plt.xlabel('Timesteps', fontsize=25)
    plt.ylabel(labels[feature], fontsize=25)
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


    