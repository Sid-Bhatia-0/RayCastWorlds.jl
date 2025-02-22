# RayCastWorlds

This package provides simple first-person 3D games that can also be used as reinforcement learning environments. It is inspired by [DeepMind Lab](https://github.com/deepmind/lab).

**Note:** This package is unregistered as of this writing and can be installed using the following command in the Julia REPL's `pkg` mode (enter `pkg` mode by pressing `]` after starting the REPL):

```
add https://github.com/Sid-Bhatia-0/RayCastWorlds.jl
```

## Table of Contents

* [Getting Started](#getting-started)
* [Notes](#notes)

[List of Environments](#list-of-environments)
1. [SingleRoom](#singleroom)

## Getting Started

```julia
import RayCastWorlds as RCW

env = RCW.SingleRoomModule.SingleRoom()

# reset the game. All environments are randomized

RCW.reset!(env)

# get names of actions that can be performed in this environment

RCW.get_action_names(env)

# perform actions in the environment

RCW.act!(env, 1) # move forward
RCW.act!(env, 2) # move backward
RCW.act!(env, 3) # turn left
RCW.act!(env, 4) # turn right

# interactively play the game
# keybindings:
# `q`: quit
# `r`: reset
# `w`: move forward
# `s`: move backward
# `a`: turn left
# `d`: turn right
# `v`: toggle top view and camera view

RCW.play!(env)

# use the RLBase API

import ReinforcementLearningBase as RLBase

# wrap a game instance from this package to create an RLBase compatible environment

rlbase_env = RCW.RLBaseEnv(env)

# perform RLBase operations on the wrapped environment

RLBase.reset!(rlbase_env)
state = RLBase.state(rlbase_env)
action_space = RLBase.action_space(rlbase_env)
reward = RLBase.reward(rlbase_env)
done = RLBase.is_terminated(rlbase_env)

rlbase_env(1) # move forward
rlbase_env(2) # move backward
rlbase_env(3) # turn left
rlbase_env(4) # turn right
```

## Notes

### RayCaster

The core raycasting algorithm is implemented in the [`RayCaster`](https://github.com/Sid-Bhatia-0/RayCaster.jl) package.

### Units

There are 4 types of units used throughout the code:
1. `wu`: Stands for world units. These are usually floating point numbers representing positions in the real world.
1. `tu`: Stands for tile units. These are integers representing positions on the tile map.
1. `pu`: Stands for pixel units. These are integers representing positions on the visualization image.
1. `au`: Stands for angle units. These are integers representing angles from 0 to `num_directions`.

The height of the tile map correponds to the x-axis of the coordinate system (often indexed with `i`) and width correponds to the y-axis (often indexed with `j`). This keeps the coordinate system right-handed.

## List of Environments

1. ### SingleRoom

    The objective of the agent is to navigate its way to the goal. When the agent tries to move into the goal tile, it receives a reward of 1 and the environment terminates.

    Camera view:

    <img src="https://user-images.githubusercontent.com/32610387/128851560-4c713b0b-cf9f-4eed-bf60-439d456ab5f6.png">

    Top view:

    <img src="https://user-images.githubusercontent.com/32610387/128851489-befdb69e-157c-48ea-b6ac-7447e8018d93.png">
