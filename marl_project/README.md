# MARL END SEM PROJECT

NAME: Mohammad Saifullah Khan  
ROLL NO.: 21169  
DEPARTMENT: EECS

# Run the Code
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install matplotlib numpy torch rich wandb
python3 maddpg.py

#To train, run on gpu.
python3 maddpg.py

#To check results for random trajectory of agents
python3 maddpg_env_random.py

# To test, 
# comment the following line in maddpg.py
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# and uncommment the following line in maddpg.py
# device = "cpu"
# rename the trained model saved on saved_models directory to tugboat_{agent_number}_actor_maddpg.pth
python3 test_maddpg.py
```

# Presentation
[Presentation](MARL Project.pdf)


