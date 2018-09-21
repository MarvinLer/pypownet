from setuptools import setup
import os
import sys


# Seeks for matpower configuration file, then put it into PYTHONPATH
# Add matpower to octave path
mp_path_config = 'matpower_path.config'
if not os.path.exists(mp_path_config):
    raise FileNotFoundError('The matpower path configuration file is not found at', mp_path_config)
with open(mp_path_config, 'r') as f:
    relative_path = f.read().splitlines()[0]

matpower_path = os.path.abspath(relative_path)
print(relative_path, matpower_path)
# Check that the matpower path does exist
if not os.path.isdir(matpower_path):
    raise FileNotFoundError('Matpower folder %s not found' % matpower_path)

# Save matpower path and replace in configuration with absolute path
sys.path.append(matpower_path)
with open(mp_path_config, 'w') as f:
    f.write(matpower_path)


# Parse required libraries file into a list of string
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()
# Replace all == with ~= for lighting dependencies
requirements = [req.replace('==', '~=') for req in requirements]

# Get the chronics files: one folder within chronics/ per case
chronics_basepath = './input/chronics'
chronics_cases = [d for d in os.listdir(chronics_basepath) if not os.path.isfile(d)]
files = []
for case in chronics_cases:
    case_path = os.path.join(chronics_basepath, case)
    # Retrieve files with relative name wrt .
    case_files = [os.path.join(case_path, f) for f in os.listdir(case_path) if f.endswith('.csv')]
    files.append(case_files)


def main():
    setup(name='pypownet',
          version='2.0.4',
          description='A Gym/keras-rl compatible environment simulating large-scale power grids.',
          author='Marvin Lerousseau',
          author_email='marvin.lerousseau@gmail.com',
          maintainer='Marvin Lerousseau',
          packages=['pypownet'],
          install_requires=requirements,
          license='LGPLv3',
          download_url='https://github.com/marvinler/pypownet',
          data_files=['matpower_path.config',
                      ('input', ['./input/reference_grid14.m', './input/reference_grid118.m', ])] +
                     [(os.path.join(chronics_basepath, case), case_files) for case, case_files in
                      zip(chronics_cases, files)],
          entry_points={'console_scripts': ['pypownet=pypownet.command_line:main']}, )


if __name__ == '__main__':
    main()
