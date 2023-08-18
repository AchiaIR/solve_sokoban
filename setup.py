from setuptools import setup, find_packages

setup(
    name='solve_sokoban',
    version='0.1',
    author='AchiaIR',
    author_email='achia.rosin19@gmail.com',
    description='A Sokoban solver using Reinforcement Deep Learning methods',
    packages=find_packages(),
    install_requires=[
        'gym',
        'tqdm',
        'numpy',
        'gdown',
        'torch',
        'pygame',
        'pyglet',
        'IPython',
        'imageio',
        'matplotlib',
        'gym_sokoban',
        'yacs >= 0.1.8',
        'imageio[ffmpeg]==2.9.0',
    ],
    package_data={
        'algorithms': ['model_free/*.py', 'model_free/__init__.py', '*.py'],
        'dnn_models': ['*py'],
        'configs': ['*.py'],
        'sokoban': ['surface/*.png', 'surface/multibox/*.png', 'surface/multiplayer/*.png', 'surface/tiny_world/*.png',
                    'surface/raw/blank_surface/*.xcf', 'surface/raw/boxes/*.xcf', 'surface/raw/player/*.xcf', '*.py'],
        'utils': ['*.py'],
        'rl_utils': ['*.py'],
    },
    data_files=[
        ('', ['solve_sokoban.py']),
    ],
    entry_points={
        'console_scripts': [
            'solve_sokoban=solve_sokoban:main',
        ],
    },
)