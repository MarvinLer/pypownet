from setuptools import setup

setup(name='gym_powernetwork',
      version='2.0.0',
      description='An OpenAI gym environment simulating the task of grid conduct through time',
      author = 'Marvin Lerousseau',
      author_email = 'marvin.lerousseau@gmail.com',
      install_requires=['gym>=0.10.5',
                        'setuptools>=38.5.1',
                        'oct2py>=4.0.6',
                        'pygame>=1.9.3',
                        'numpy>=1.14.0',
                        'scipy==1.1.0',]
)