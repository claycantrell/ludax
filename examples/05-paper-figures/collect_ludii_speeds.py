import argparse
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use for parallel processing")
parser.add_argument("--action_cap", type=int, default=200, help="Maximum number of actions to take in each game")
parser.add_argument("--tensor_playouts", action="store_true", help="Whether to use tensor playouts (if not set, will use regular playouts)")

args = parser.parse_args()

GAMES = [
    "Connect Four",
    "Connect6",
    "Dai Hasami Shogi",
    "English Draughts",
    "Gomoku",
    "Havannah",
    "HopThrough",
    "Hex",
    "Pente",
    "Reversi",
    "Tic-Tac-Toe",
    "Wolf and Sheep",
    "Yavalath",
    "Yavalax",
]

if args.tensor_playouts:
    print("Using tensor playouts!")
    time_command = "--time-tensor-playouts"
    playout_str = "tensor_playouts"

else:
    print("Using regular playouts!")
    time_command = "--time-playouts"
    playout_str = "regular_playouts"

game_list_str = ' '.join([f'"/{game}.lud"' for game in GAMES])

output_path = f"data/ludii_speeds_{playout_str}_{args.num_threads}_threads.csv"
command = f"java -jar Ludii.jar {time_command} --num-threads {args.num_threads} --game-names {game_list_str} --export-csv {output_path} --playout-action-cap {args.action_cap} "

print(f'Running command: "{command}"')
os.system(command)
