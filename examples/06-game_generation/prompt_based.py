import os
import regex as re
import json
import datetime

from cerebras.cloud.sdk import Cerebras
from ludax import grammar as ludax_grammar, LudaxEnvironment
from ludax import games

from gavel_eval import gavel_metrics, initialize
from random_games import simple_eval

from coolname import generate


def construct_game_description() -> str:
    """Construct a game description in Ludax format."""
    name = " ".join(word.capitalize() for word in generate(3))
    content = f"""
        Invent simple rules for a novel two player abstract strategy game called {name}. Implement it in the ludax language. You will find attached the ludax's grammar as well as a few examples of games implemented in ludax. Start by implementing a simplified version of your rules, and then incrementally add rules that are harder to express in ludax. At each step, make sure you write a compilable game according to ludax's grammar.

        grammar.lark:\n{ludax_grammar}


        example games:
        """

    for game in games.__all__:
        content += f"\n\n{getattr(games, game)}\n"

    return content


cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
def ask_cerebras(content, model, temperature) -> str:
    """Prompt the Cerebras API to generate a game description."""
    chat_completion = cerebras_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=model,
        temperature=temperature
    )

    return chat_completion.choices[0].message.content



def strip_comments_and_nonascii(src: str) -> str:
    # 1) Remove // and ;… end-of-line comments, but *ignore* semicolons inside quoted strings.
    no_comments = re.sub(
        r'"(?:\\.|[^"\\])*"(*SKIP)(*F)|;[^\r\n]*|//[^\r\n]*',
        '',
        src,
        flags=re.MULTILINE
    )
    # 2) Remove all non-ASCII characters everywhere (including inside strings).
    cleaned = re.sub(r'[^\p{ASCII}]', '', no_comments)
    # Optional: trim trailing whitespace on each line and strip leading/trailing blank lines.
    cleaned = re.sub(r'[ \t]+(?=\r?$)', '', cleaned, flags=re.MULTILINE).strip()
    return cleaned


pattern = re.compile(r"\(game\b(?:[^()]+|(?P<par>\((?:[^()]+|(?P>par))*\)))*\)")
def extract_game_descriptions(text: str) -> list[str]:
    games = [m.group(0) for m in pattern.finditer(text)]
    return [strip_comments_and_nonascii(game) for game in games if game.strip()]



def main():
    # model = "llama-4-scout-17b-16e-instruct"
    model = "gpt-oss-120b"
    # model = "qwen-3-235b-a22b-thinking-2507"
    temperature = 0.2
    iterations = 100

    generated_games = {
        "model": model,
        "temperature": temperature,
        "time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "working_games": 0,
        "total_games": iterations,
        "requests": [],
        "prompt": construct_game_description(),
    }
    for i in range(iterations):
        print(f"\n\nIteration {i + 1}", "-" * 40)
        content = construct_game_description()
        response = ask_cerebras(content, model=model, temperature=temperature)

        game_evals = []
        game_descriptions = extract_game_descriptions(response)

        for game_description in game_descriptions:
            print(f"\nGame description:\n{game_description}")

            try:
                game_eval = simple_eval(game_description)
            except Exception as e:
                print(f"Error evaluating game: {e}")
                game_eval = "Exception"

            print(f"Game evaluation: {game_eval}")
            game_evals.append({
                "description": game_description,
                "evaluation": game_eval,
            })

        if any(eval["evaluation"] == "Playable" for eval in game_evals):
            print("Found a playable game!")
            generated_games["working_games"] += 1

        generated_games["requests"].append({
            "response": response,
            "games": game_evals,
        })

        with open(f"./output/{model}-{generated_games['time']}.json", "w") as f:
            json.dump(generated_games, f, indent=2)




if __name__ == "__main__":
        main()








