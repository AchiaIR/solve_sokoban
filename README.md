<div align="center">
    <h1>Solve Sokoban with Deep Reinforcement Learning</h1>
<img src="https://github.com/AchiaIR/solve_sokoban/" width="500" height="500" />
</div>

## Table of Contents

- <a href="#about-the-project">About The Project</a>
- <a href="#overview">Overview</a>
- <a href="#project-content-description">Project Content Description</a>
  - <a href="#python-content-description">Python Content Description</a>
  - <a href="#other-supporting-content-description">Other Supporting Content Description</a>
- <a href="#getting-started">Getting Started</a>
  - <a href="#prerequisites">Prerequisites</a>
  - <a href="#installation">Installation</a>
    - <a href="#windows">Windows</a>
    - <a href="#linux-and-mac">Linux and Mac</a>
  - <a href="#usage">Usage</a>
    - <a href="#windows">Windows</a>
    - <a href="#linux-and-mac">Linux and Mac</a>
- <a href="#advanced-usage">Advanced Usage</a>
- <a href="#acknowledgments">Acknowledgments</a>

<span> ───────────────────────────────────────────── </span>

<a name="about-the-project"></a>
<h2> $\color{red}{ About \ The \ Project }$ </h2>

This project is a showcase of how the Sokoban game, a difficult
transportation puzzle where boxes are moved to storage locations, can be solved
using Reinforcement Learning approach. 
Here we use Deep Q-Network, A model-free algorithm, Q-Network learns by trial and error. 
It gauges the value of an action in a particular state and refines its strategy over time, making it adaptable and dynamic.
Deep Q-Network means making use of deep neural network as "Q-table", the way the algorithm takes decisions. 

<span> ───────────────────────────────────────────── </span>

<a name="overview"></a>
<h2> $\color{red}{Overview }$ </h2>

Explain about the 2 parts of the project  - the Sokoban game, and the Deep Q-Network algorithm:
* **Sokoban:** Sokoban (meaning 'warehouse keeper’) is a Japanese puzzle video game genre in which the 
player pushes crates or boxes around in a warehouse, trying to get them to storage locations.
The game is a transportation puzzle, where the player has to push all boxes in the room on the 
storage locations/ targets. The possibility of making irreversible mistakes makes these puzzles so 
challenging especially for Reinforcement Learning algorithms, which mostly lack the ability to 
think ahead. The game is played on a board of squares, where each square is a floor or a wall. Some floor 
squares contain boxes, and some floor squares are marked as storage locations.
The player is confined to the board and may move horizontally or vertically onto empty squares 
(never through walls or boxes). The player can move a box by walking up to it and push it to the 
square beyond. Boxes cannot be pulled, and they cannot be pushed to squares with walls or other 
boxes. The number of boxes equals the number of storage locations. The puzzle is solved when 
all boxes are placed at storage locations.

* **deep Q-network:** deep Q-network (DQN) is a type of artificial neural network that learns to approximate the optimal Q-value function in a reinforcement learning problem.
The Q-value function maps states to action values, indicating the expected reward that an agent can achieve by taking a particular action in a given state.
In DQN, the network receives a state as input and outputs the Q-values for
all possible actions. The network is trained using a combination of supervised
and reinforcement learning methods, where the loss function is defined as the
difference between the predicted Q-values and the actual rewards obtained from
the environment. The DQN algorithm involves iteratively updating the network parameters
using batches of experiences sampled from the replay buffer.

<span> ───────────────────────────────────────────── </span>

<a name="project-content-description"></a>
<h2> $\color{red}{ Project \ Content \ Description }$ </h2>

<a name="python-content-description"></a>
<h3> $\color{lime}{Python \ Content  \ Description}$ </h3>

* **algorithms:** A folder contains the implementation of the RL algorithms, with a folder: 
    - model free 
* **configs:** A folder contains the files to do with Sokoban and algorithms configurations.
* **ednn_models:** A folder contains the files define the deep neural networks employed by the algorithms.
* **rl_utils:** A folder contains the implementation of replay buffer and the tools used for rl.
* **utils:** A folder contains the display file, and the file that gets the sokoban folder (to use once).
* **sokoban:** A folder contains the the files consist of the sokoban game.
* **solve_sokoban.py:** The main file - the file which runs the project.
* **setup.py:** the file to use in installation.

<a name="other-supporting-content-description"></a>
<h3>$\color{lime}{Other \ supporting \ Content \ Description}$</h3>

* **comfig.yaml:** A file that defines the configuration to use
* **requirements.txt:** A file that defines the libraris to install for running the project
* **run.bat:** A file which runs the project from command  - windows
* **run.sh:** A file which runs the project from command  - linux / mac

<span> ───────────────────────────────────────────── </span>

<a name="getting-started"></a>
<h2> $\color{red}{ Getting \ Started }$ </h2>

<a name="prerequisites"></a>
<h3>$\color{lime}{Prerequisites}$</h3>

* pytghon 3.x installed on your machine
* Pip (Python package installer)
* A video tool installed (for example vlc) 

<a name="installation"></a>
<h3>$\color{lime}{Installation}$</h3>

<a name="windows"></a>
* <h3>$\color{cyan}{Windows:}$</h3>

1. Clone this repository or download and extract the ZIP file:

   `git clone https://github.com/AchiaIR/solve_sokoban.git`

2. Navigate to the directory where you cloned or extracted the project:

   `cd solve_sokoban`
   
3. Install the necessary dependencies:

   `pip install -r requirements.txt`

   or:

   `pip install .`

   or:

   `python setup.py install` 

<a name="linux-and-mac"></a>   
* <h3>$\color{cyan}{Linux \ and \ Mac:}$</h3>

1. Clone this repository or download and extract the ZIP file:

   `git clone https://github.com/AchiaIR/solve_sokoban.git`

2. Navigate to the directory where you cloned or extracted the project:

   `cd solve_sokoban`
   
3. Install the necessary dependencies:

   `pip3 install -r requirements.txt`

   or:

   `pip3 install .`

   or:
   
   `pip3 install setuptools`
   
   `python3 setup.py install`

<a name="usage"></a>
<h3>$\color{lime}{Usage}$</h3>

<a name="windows"></a>
* <h3>$\color{cyan}{Windows:}$</h3>

Run the run.bat script:

`run`

you can play with the number of boxes:

`run 2`

<a name="linux-and-mac"></a>
* <h3>$\color{cyan}{Linux \ and \ Mac:}$</h3>

1. Make the script executable (only need to do this once):

   `chmod +x ./run.sh`

2. Run the file.sh script:

   `./run.sh`

   you can play with the number of boxes:

   `run 2`

<span> ───────────────────────────────────────────── </span>

<a name="advanced-usage"></a>
<h2>$\color{red}{Advanced \ Usage}$</h2>

Since the algorithm is based on training a deep neural netowrk, running it with a random environement, where it creates a 
new game structure each time, means it will have to train it from scratch in each run. This may take a long time on CPU. Therefor 
I trained networks for a constant seed (seed=2), and upload the pth files here, so when one runs it it will create the same game structure and will solve it using the 
pre-trained networks. 

If you wish to train it yourself and explore it further, you can uncomment the function SetAlgorithmWithTraining in solve_sokoban.py, and also the call to that finction in main function,
and run it with traininig from scratch. You better have a GPU installed f you choose to do so. 

You also may play with the paramteres, explor the best parameters for fast convergence, and maybe chnaging the reward function and see how it affects the solving abilities. 
Another aspect to test is the number of episodes and maximum steps in each episode  -  the less the better, 
on a condition the convergence percentage (how many times you run it succesfully on a new structure) is high enough.

I also add here the google colab notebook for an easy exploration.

<span> ───────────────────────────────────────────── </span>

<a name="acknowledgments"></a>
<h2>$\color{red}{Acknowledgments}$</h2>

Based on a project in Reinforcement Learning course, Reichman University




