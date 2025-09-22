from gavel_eval import evaluate_game
from src.ludax import LudaxEnvironment
import os
import json



if __name__ == "__main__":
    # Load previously generated games
    file_name = "gpt-oss-120b-2025-08-29_18-58-14.json"
    with open("output/" + file_name, "r") as f:
        generated_games = json.load(f)


    # Re-evaluate all games and update their scores
    for i in range(len(generated_games["requests"])):
        for j in range(len(generated_games["requests"][i]["games"])):

            print(f"\n\nRe-evaluating game {i}-{j}...")

            game_entry = generated_games["requests"][i]["games"][j]
            description = game_entry["description"]
            try:
                gavel_score, score_breakdown = evaluate_game(description)
                print(f"Re-evaluated game {i}-{j}: {gavel_score}, Breakdown: {score_breakdown}")
                game_entry["gavel_score"] = float(gavel_score)
                game_entry["gavel_breakdown"] = [float(x) for x in score_breakdown]
            except Exception as e:
                print(f"Error re-evaluating game {i}-{j}: {e}")
                game_entry["evaluation"] = "Exception"

            # Save updated games back to the file
            with open("output/scored_" + file_name, "w") as f:
                json.dump(generated_games, f, indent=2)

    print("Re-evaluation complete. Updated scores saved to generated_games.json.")