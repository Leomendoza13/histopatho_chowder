"""Script to Ensemble 50 Models"""

import subprocess

# Iterate through 50 models
for i in range(50):
    # Define the filename for the weights of each model
    file_name = "weights" + str(i) + ".pth"

    # Execute training script for each model, saving weights with corresponding filename
    result = subprocess.run(["python", "train.py", "--save", file_name], check=True)

# After training all models, execute ensemble script to combine them
result = subprocess.run(["python", "ensemble.py"], check=True)
