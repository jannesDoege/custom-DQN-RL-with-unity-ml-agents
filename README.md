# custom-DDQN-RL-with-unity-ml-agents

The folder "unity-project" can be opened in Unity as a project. 

"unity-project/Assets/scripts/MoveToGoalAgent.cs" is the script that contains the logic on how the agent interacts with the environment and which information get transmitted to the "python side".

All of the python code containing the DDQN algorithm is in "unity-project/python". 

To train the model run "unity-project/python/train.py" and run the unity-scene when the instruction to do so is printed in the terminal.

The hyperparamaters in "unity-project/python/hyperparams.py" are not optimized - tuning them and the rewards specified in "unity-project/Assets/scripts/MoveToGoalAgent.cs" could improve training significantly.

I am using python 3.8.8 and my Unity version is 2019.4.16f1 (if you are using a newer unity version you can update the project to a newer version - there shouldn't be any issues)
