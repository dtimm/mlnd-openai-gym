# mlnd-openai-gym
Udacity MLND Capstone: Classic Control Problems with Normalized Advantage Functions

Using Normalize Advantage Functions from http://arxiv.org/abs/1603.00748 to
solve OpenAI Gym Classic Control environments.

Dependencies:
tensorflow r0.10
```
pip install gym numpy
```

To run the simulation, you can use the following commands:
```
python main.py
python main.py -e CartPole-v0
python main.py -e MountainCar-v0
python main.py -e Acrobot-v0
```
