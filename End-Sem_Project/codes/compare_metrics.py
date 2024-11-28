import pandas as pd
import matplotlib.pyplot as plt
mappo_results = pd.read_csv("test_logs_mappo_cahnge/MAPPO_scores.csv")
iql_results = pd.read_csv("test_logs_iql_change/IQL_scores.csv")

def compute_metrics(data):
    total_episodes = len(data)
    win_rate_team_1 = (data["Team_1_Total_Wins"].iloc[-1] / total_episodes) * 100
    win_rate_team_2 = (data["Team_2_Total_Wins"].iloc[-1] / total_episodes) * 100
    draw_rate = (data["Draws"].iloc[-1] / total_episodes) * 100
    avg_score_team_1 = data["Team_1_Score"].mean()
    avg_score_team_2 = data["Team_2_Score"].mean()
    avg_score_diff = (data["Team_1_Score"] - data["Team_2_Score"]).mean()

    return {
        "Win Rate Team 1": win_rate_team_1,
        "Win Rate Team 2": win_rate_team_2,
        "Draw Rate": draw_rate,
        "Average Score Team 1": avg_score_team_1,
        "Average Score Team 2": avg_score_team_2,
        "Average Score Difference": avg_score_diff,
    }

mappo_metrics = compute_metrics(mappo_results)
iql_metrics = compute_metrics(iql_results)
print("#################################")
print("\n\n")
print("MAPPO Metrics:")
print(mappo_metrics)
print("#################################")
print("\n\n")
print("IQL Metrics:")
print(iql_metrics)
print("\n\n")
print("#################################")

metrics_comparison = pd.DataFrame([mappo_metrics, iql_metrics], index=["MAPPO", "IQL"])
print(metrics_comparison)

fig, ax = plt.subplots(2, 1, figsize=(8, 10))
metrics_comparison[["Win Rate Team 1", "Win Rate Team 2", "Draw Rate"]].plot(kind="bar", ax=ax[0])
ax[0].set_title("Win and Draw Rates")
ax[0].set_ylabel("Percentage")
ax[0].set_xticklabels(metrics_comparison.index, rotation=0)

metrics_comparison[["Average Score Team 1", "Average Score Team 2"]].plot(kind="bar", ax=ax[1])
ax[1].set_title("Average Scores")
ax[1].set_ylabel("Score")
ax[1].set_xticklabels(metrics_comparison.index, rotation=0)

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(mappo_results["Episode"], mappo_results["Team_1_Score"], label="MAPPO Team 1", linestyle='--', color='cyan')
plt.plot(mappo_results["Episode"], mappo_results["Team_2_Score"], label="MAPPO Team 2", linestyle='--', color='green')
plt.plot(iql_results["Episode"], iql_results["Team_1_Score"], label="IQL Team 1", linestyle='-', color='cyan')
plt.plot(iql_results["Episode"], iql_results["Team_2_Score"], label="IQL Team 2", linestyle='-', color='green')
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Scores Over Episodes")
plt.legend()
plt.show()
