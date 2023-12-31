import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from colorama import Fore, Back, Style
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, width=10, height=10, render_mode="console"):
        super(GoLeftEnv, self).__init__()
        self.render_mode = render_mode
        self.wall = np.array([[]])
        # Size of the 1D-grid
        self.width = width
        self.height = height
        # Initialize the agent at the right of the grid
        self.agent_location = np.array([1,1])
        self.reward = 0
        self.total_time = 0
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.width - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.width - 1, shape=(2,), dtype=int),
            }
        )
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        # Create a set of randomized walls and then use that for the whole training set
        
        self.wall=np.array([])
        rng = np.random.default_rng()
        size = self.width*self.height
        #percent_wall = rng.integers(1, (size //2))
        percent_wall = size //3
        self.rand_wall_cells=[]
        for i in range(percent_wall):
            j=rng.integers(0,(self.width*self.height))
            while j in self.rand_wall_cells:
                j=rng.integers(0,(self.width*self.height))
            self.rand_wall_cells.append(j)
        self.rand_wall_cells = np.array(self.rand_wall_cells)
        
        total =0
        for i in range(self.height):
            for j in range(self.width):
                if i == 0 or j == 0 or i == self.height -1 or j == self.width - 1:
                    
                    self.wall=np.append(self.wall, [i,j])
                    total+=1
                
                elif  self.width*i+j in self.rand_wall_cells:
                    
                    self.wall=np.append(self.wall, [i,j])
                    total+=1
                 
        self.wall=self.wall.reshape(total, 2)
        
        self.visited = []
    def _get_obs(self):
        return {"agent": self.agent_pos, "target": self._target_location}
    def _get_info(self):
        info ={"distance": np.linalg.norm(
                self.agent_pos - self._target_location, ord=1)}
        for i in self.wall:
            _id = tuple(i)
            info[_id]= np.linalg.norm(
            self.agent_pos - i, ord=1)
        info["visited"]=self.visited
        return info
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array([3,3])
        self._target_location = np.array([self.width//2, self.height//2])
        
        # time reward reset
        self.reward = 0
        self.time_total = 0
        
        #reset its path
        self.visited = []
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        
        #create a randomized set of interior walls
        
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info  # empty info dict

    def step(self, action):
        self.time_total +=1
        reward=0
        direction = self._action_to_direction[action]  
        truncated = False
        agent_new_pos = self.agent_pos + direction
        if (self.wall==agent_new_pos).all(1).any():
            pass
        else:
            self.agent_pos = agent_new_pos
        # want to encourage the agent to move to new tiles over visiting old ones 
        """
        current_cell = tuple(self.agent_pos)
        self.visited.append(current_cell)
        if self.visited.count(current_cell) <= 2:
            reward+=5
        else:
            reward -= self.visited.count(current_cell)
        """
        
        
        terminated = np.array_equal(self.agent_pos, self._target_location)
        if terminated == True:
            reward +=500# Binary sparse rewards
        if self.time_total >= 500:
            truncated = True
        
        observation = self._get_obs()
        

        # Optionally we can pass additional info, we are not using that for now
        info = self._get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            for y in range(self.height):
                for x in range(self.width):
                    if np.array_equal([x,y], self.agent_pos):
                        print(Fore.MAGENTA + "X", Style.RESET_ALL, end="")
                    elif np.array_equal([x,y], self._target_location):
                        print(Fore.GREEN + "0", Style.RESET_ALL, end="")
                    elif np.any(np.all([x,y] == self.wall, axis=1)):
                        print("1", end=" ")
                    else:
                        print("0", end=" ")
                print("")
    def close(self):
        pass
 
     



vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(width=15, height =15))
model = A2C("MultiInputPolicy", vec_env, verbose=1)
obs=vec_env.reset()
vec_env.render()
eval_env = vec_env
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

model.learn(50000)


obs = vec_env.reset()
n_steps = 500
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
