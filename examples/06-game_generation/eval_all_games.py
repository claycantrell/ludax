import jax
import jax.numpy as jnp
import json
import os

from gavel_eval import evaluate_game
from ludax import games

def print_stats(gavel_scores):
    gavel_scores = jnp.array(gavel_scores)
    print(f"Average Gavel score: {jnp.nanmean(gavel_scores[:, 0])} ± {jnp.nanstd(gavel_scores[:, 0])}")
    print(f"Average Balance: {jnp.nanmean(gavel_scores[:, 1])} ± {jnp.nanstd(gavel_scores[:, 1])}")
    print(f"Average Decisiveness: {jnp.nanmean(gavel_scores[:, 2])} ± {jnp.nanstd(gavel_scores[:, 2])}")
    print(f"Average Completion: {jnp.nanmean(gavel_scores[:, 3])} ± {jnp.nanstd(gavel_scores[:, 3])}")
    print(f"Average Agency: {jnp.nanmean(gavel_scores[:, 4])} ± {jnp.nanstd(gavel_scores[:, 4])}")
    print(f"Average Coverage: {jnp.nanmean(gavel_scores[:, 5])} ± {jnp.nanstd(gavel_scores[:, 5])}")
    print(f"Average Strategic Depth: {jnp.nanmean(gavel_scores[:, 6])} ± {jnp.nanstd(gavel_scores[:, 6])}")

def eval_ludax_games():
    gavel_scores = []
    compilable = 0
    total = len(games.__all__)
    for game in games.__all__:
        game_str = getattr(games, game)
        print(f"\n\nEvaluating game: {game}")
        mean, metrics = evaluate_game(game_str)
        assert metrics is not None

        print(f"Gavel score: {mean}, metrics: {metrics}")
        gavel_scores.append((mean, *metrics))
        compilable += 1

    print(f"Generated {total} games with:")
    print(f"Compilable: {compilable}/{total}")
    if compilable > 0:
        print_stats(gavel_scores)







def from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    gavel_scores = []
    compilable = 0
    interesting = 0
    total = 0
    for request in data.get("requests", []):
        best_scores = None
        total += 1

        for game_entry in request.get("games", []):

            if "gavel_score" in game_entry and "gavel_breakdown" in game_entry:
                mean = float(game_entry["gavel_score"])
                metrics = [float(x) for x in game_entry["gavel_breakdown"]]
                assert len(metrics) == 6

                print(f"Gavel score: {mean}, metrics: {metrics}")

                # # Correct Strategic Depth and mean calculation
                # metrics[5] = 2 * metrics[5] - 1 # Convert from [0.5, 1] to [0, 1]
                # mean = 6 / sum(1 / m if m > 0 else jnp.inf for m in metrics)

                if best_scores is None or mean > best_scores[0]:
                    best_scores = (mean, *metrics)

        if best_scores:
            compilable += 1
            gavel_scores.append(best_scores)

            if best_scores[0] > 0.5:
                interesting += 1

        print(f"Generated {total} games with:")
        print(f"Compilable: {compilable}/{total}")
        print(f"Interesting: {interesting}/{total}")
        if compilable > 0:
            gavel_scores2 = jnp.array(gavel_scores)
            print(f"Average Gavel score: {jnp.nanmean(gavel_scores2[:, 0])} ± {jnp.nanstd(gavel_scores2[:, 0])}")
            print(f"Average Balance: {jnp.nanmean(gavel_scores2[:, 1])} ± {jnp.nanstd(gavel_scores2[:, 1])}")
            print(f"Average Decisiveness: {jnp.nanmean(gavel_scores2[:, 2])} ± {jnp.nanstd(gavel_scores2[:, 2])}")
            print(f"Average Completion: {jnp.nanmean(gavel_scores2[:, 3])} ± {jnp.nanstd(gavel_scores2[:, 3])}")
            print(f"Average Agency: {jnp.nanmean(gavel_scores2[:, 4])} ± {jnp.nanstd(gavel_scores2[:, 4])}")
            print(f"Average Coverage: {jnp.nanmean(gavel_scores2[:, 5])} ± {jnp.nanstd(gavel_scores2[:, 5])}")
            print(f"Average Strategic Depth: {jnp.nanmean(gavel_scores2[:, 6])} ± {jnp.nanstd(gavel_scores2[:, 6])}")



def print_sorted(input_path):
    # Print all games sorted by their Strategic Depth, print their descriptions and scores
    with open(input_path, "r") as f:
        data = json.load(f)

    games_list = []
    for request in data.get("requests", []):
        for game_entry in request.get("games", []):
            if "gavel_score" in game_entry and "gavel_breakdown" in game_entry:
                mean = float(game_entry["gavel_score"])
                metrics = [float(x) for x in game_entry["gavel_breakdown"]]
                assert len(metrics) == 6

                games_list.append((game_entry.get("description", ""), mean, *metrics))

    games_list.sort(key=lambda x: x[-1], reverse=True)
    for score in games_list:
        print(f"GAVEL Score: {score[1]:.4f}, Balance: {score[2]:.4f}, Decisiveness: {score[3]:.4f}, "
              f"Completion: {score[4]:.4f}, Agency: {score[5]:.4f}, Coverage: {score[6]:.4f}, "
              f"Strategic Depth: {score[7]:.4f}\nDescription:\n{score[0]}\n")


if __name__ == "__main__":
    # eval_ludax_games()

    # file_name = "scored_gpt-oss-120b-2025-08-29_18-58-14.json"
    file_name = "scored_llama-4-scout-17b-16e-instruct-2025-08-29_18-42-24.json"
    input_path = os.path.join("output", file_name)
    from_file(input_path)

    # print_sorted(input_path)


