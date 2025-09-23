from gavel_eval import evaluate_game
import os
import json
import time
from multiprocessing import Process, Queue

TIMEOUT_SECONDS = 10 * 60  # 10 minutes


def _worker(q: Queue, description: str):
    """Run evaluate_game in a separate process and put the result on the queue."""
    try:
        gavel_score, score_breakdown = evaluate_game(description)
        q.put({
            "status": "ok",
            "gavel_score": float(gavel_score),
            "gavel_breakdown": [float(x) for x in score_breakdown],
        })
    except Exception as e:
        q.put({
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        })


def evaluate_with_timeout(description: str, timeout_s: int = TIMEOUT_SECONDS):
    """
    Launch a new process to evaluate a single game.
    If it exceeds timeout_s, terminate the process.
    Returns a dict with status, duration_seconds, and (if ok) scores.
    """
    q = Queue()
    p = Process(target=_worker, args=(q, description))
    start = time.monotonic()
    p.start()
    p.join(timeout_s)
    duration = time.monotonic() - start

    if p.is_alive():
        # Timed out: terminate the child and report timeout
        try:
            p.terminate()
        finally:
            p.join(5)  # give it a moment to exit
        try:
            q.close()
        except Exception:
            pass
        return {"status": "timeout", "duration_seconds": duration}

    # Process finished in time; try to get its result
    result = None
    try:
        if not q.empty():
            result = q.get_nowait()
    except Exception:
        result = None
    finally:
        try:
            q.close()
        except Exception:
            pass

    if not result:
        return {"status": "error", "error": "No result from subprocess.", "duration_seconds": duration}

    result["duration_seconds"] = duration
    return result


if __name__ == "__main__":
    # Load previously generated games
    file_name = "gpt-oss-120b-2025-08-29_18-58-14.json"
    input_path = os.path.join("output", file_name)
    output_path = os.path.join("output", "scored_" + file_name)

    with open(input_path, "r") as f:
        generated_games = json.load(f)

    # Re-evaluate all games and update their scores
    for i in range(len(generated_games["requests"])):
        for j in range(len(generated_games["requests"][i]["games"])):

            print(f"\n\nRe-evaluating game {i}-{j}...")
            game_entry = generated_games["requests"][i]["games"][j]
            description = game_entry.get("description", "")

            result = evaluate_with_timeout(description, TIMEOUT_SECONDS)

            # Store the duration regardless of outcome
            game_entry["evaluation_duration_seconds"] = round(float(result.get("duration_seconds", 0.0)), 6)

            if result["status"] == "ok":
                print(f"Re-evaluated game {i}-{j}: {result['gavel_score']}, "
                      f"Breakdown: {result['gavel_breakdown']}")
                game_entry["gavel_score"] = result["gavel_score"]
                game_entry["gavel_breakdown"] = result["gavel_breakdown"]
                game_entry["evaluation"] = "Success"
            elif result["status"] == "timeout":
                print(f"Timeout re-evaluating game {i}-{j} after {TIMEOUT_SECONDS} seconds.")
                game_entry["evaluation"] = "Timeout"
            else:
                print(f"Error re-evaluating game {i}-{j}: {result.get('error', 'Unknown error')}")
                game_entry["evaluation"] = "Exception"
                game_entry["evaluation_error"] = result.get("error", "Unknown error")

            # Save updated games back to the file after each evaluation
            with open(output_path, "w") as f:
                json.dump(generated_games, f, indent=2)

    print(f"Re-evaluation complete. Updated scores saved to {output_path}.")
