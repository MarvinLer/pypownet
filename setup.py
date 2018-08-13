from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()
# Replace all == with ~= for lighting dependencies
requirements = [req.replace('==', '~=') for req in requirements]

setup(name='pypownet',
      version='2.0.0',
      description='An OpenAI Gym environment implemented on top of a power network simulator',
      author = 'Marvin Lerousseau',
      author_email = 'marvin.lerousseau@gmail.com',
      packages=['pypownet',],
      install_requires=requirements,
)