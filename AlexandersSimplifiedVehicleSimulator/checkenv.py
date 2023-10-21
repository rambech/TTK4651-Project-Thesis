from stable_baselines3.common.env_checker import check_env
from rl.env import SimpleEnv
from vehicle import Otter

vehicle = Otter()
env = SimpleEnv(vehicle)

# It will check your custom environment and output additional warnings if needed
check_env(env)
