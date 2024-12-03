RUN_NAME = 'with_variability' # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all models in this folder WILL be overwritten.

OUT_NAME = f'runs/{RUN_NAME}'

import os 
import sys
import subprocess
# Set working directory to root of repo
current_path = os.getcwd()
# Identify if path has 'molanalysis' as a folder in it
if 'Petreanu_MEI_generation' in current_path:
    # If so, set the path to the root of the repo
    current_path = current_path.split('Petreanu_MEI_generation')[0] + 'Petreanu_MEI_generation'
else:
    raise FileNotFoundError(
        f'This needs to be run somewhere from within the Petreanu_MEI_generation folder, not {current_path}')
os.chdir(current_path)
sys.path.append(current_path)

for i in range(5):
    subprocess.run(['python', 'scripts/train_model.py', '-m', f'config_m4_ens{i}', '-l', OUT_NAME, '-dl', f'{OUT_NAME}/data'])