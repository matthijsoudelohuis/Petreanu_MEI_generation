import subprocess
import os
import sys

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

# list of relative locations, in order, of the python scripts to be run
files_list = [
    # "../molanalysis/MEI_generation/IM_dataconversion.py",
    'IM_dataconversion.py',
    "1. preprocess_data.py",
    "2. train_models.py",
    "3. evaluate.py",
    "4. generateMEIs.py"
    ]

current_path = os.getcwd()

# Run each script in the list
for file in files_list:
    print(f'Running {file}')
    if '../' in file:
        os.chdir(os.path.join('..', file.split('/')[1]))
        print(f'Changed directory to {os.getcwd()}')
    else:
        os.chdir(current_path)
        print(f'Changed directory to {os.getcwd()}')
    result = subprocess.run(['python', file],capture_output = True,text = True)
    print(result.stdout)
    print(result.stderr)


# MOL comments: 
# - Handle errors in running scripts, show output and warn user 
# (e.g if file or dir not found, etc.)
# - need torch compiled with CUDA enabled