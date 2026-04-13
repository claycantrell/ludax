import argparse
import datetime
import os
import subprocess

# command = 'java -jar Ludii.jar --eval-games --AIName "UCT" --iteration-limit 2000 --game-names "~/projects/ludax/examples/07-ludii-comparison/tic_tac_toe_mod.lud" --thinkTime -1 --numTrials 10'
command = 'java -jar Ludii.jar --eval-games --AIName "UCT" --iteration-limit 2000 --game-names "./tic_tac_toe_mod.lud" --thinkTime -1 --numTrials 20'


print(f'Running command: "{command}"')

# Run the command and capture the stdout and stderr
result = subprocess.run(command, shell=True, capture_output=True, text=True)
if result.stderr:
    print(result.stderr)
    exit()

# Get the P1 and P2 winrate
p1_winrate = [line for line in result.stdout.splitlines() if line.startswith("Player 1 win rate:")][0]
p2_winrate = [line for line in result.stdout.splitlines() if line.startswith("Player 2 win rate:")][0]

print(p1_winrate)
print(p2_winrate)
print(result.stdout)