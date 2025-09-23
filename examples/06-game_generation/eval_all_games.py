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
    total = 0
    for request in data.get("requests", []):
        for game_entry in request.get("games", []):
            total += 1

            if "gavel_score" in game_entry and "gavel_breakdown" in game_entry:
                mean = float(game_entry["gavel_score"])
                metrics = [float(x) for x in game_entry["gavel_breakdown"]]
                assert len(metrics) == 6

                print(f"Gavel score: {mean}, metrics: {metrics}")
                gavel_scores.append((mean, *metrics))
                compilable += 1


if __name__ == "__main__":
    eval_ludax_games()

    # file_name = "gpt-oss-120b-2025-08-29_18-58-14.json"
    # input_path = os.path.join("output", file_name)
    # from_file(input_path)


