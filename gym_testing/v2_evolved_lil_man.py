import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from colorama import Fore, Back, Style
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env


# I want to add a tag element - which means I have to get rid of the end option

    


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

    def __init__(self, width=10, height=10, enemy_width=2, enemy_height=8, render_mode="console"):
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
        self.observation_space = spaces.Box(low=0, high=self.width-1,
                                        shape=(4,), dtype=np.int64)
        
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
        percent_wall = rng.integers(1, (size //2))
        #percent_wall = size //3
        self.rand_wall_cells=[]
        """
        for i in range(percent_wall):
            j=rng.integers(0,(self.width*self.height))
            while j in self.rand_wall_cells:
                j=rng.integers(0,(self.width*self.height))
            self.rand_wall_cells.append(j)
        self.rand_wall_cells = np.array(self.rand_wall_cells)
        """
        
        # Just create some nice solid walls
        wall_min = self.height//4
        wall_max = 3*wall_min
        total =0
        for i in range(self.height):
            for j in range(self.width):
                if i == 0 or j == 0 or i == self.height -1 or j == self.width - 1:
                    
                    self.wall=np.append(self.wall, [j,i])
                    total+=1
                """
                elif  self.width*i+j in self.rand_wall_cells:
                    
                    self.wall=np.append(self.wall, [i,j])
                    total+=1
                
                if i in range(wall_min,wall_max) and ( j == self.width//3 or j== 2*self.width//3):
                   self.wall=np.append(self.wall, [j,i])
                   total+=1 
                """
        self.wall=self.wall.reshape(total, 2)
        
        self.visited = []
        self.seen=[]
        
        self.TagList =[]
        self.enemy_height=enemy_height
        self.enemy_width=enemy_width
        self.TagList=self.MakeTag(enemy_width,enemy_height,self.TagList)
        
    def _get_obs(self):
        return_this = np.array([self.agent_pos[0], self.agent_pos[1]])
        return_this1=[]
        for enemy in self.TagList:
            return_this = np.append(return_this, enemy[0])
            return_this = np.append(return_this, enemy[1])
        for i in return_this:
            return_this1.append(i)
        return_this=np.array(return_this1)
        return  return_this
    def _get_info(self):
        info ={}
        return info
    
    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array([3,3])
        
        
        # time reward reset
        
        self.time_total = 0
        

        self.visited = [[3,3]]
        self.seen=[[3,3]]

        
        
        observation = self._get_obs()
        info = self._get_info()
        self.TagList =[]
        self.TagList=self.MakeTag(self.enemy_width,self.enemy_height,self.TagList)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info  # empty info dict

    def step(self, action):
        self.time_total +=1
        reward=0
        terminated = False
        
        #have to have the enemies move prior to action 4 - otherwise it learns it can just take action 4 50 times in a row and not get hit
        """
        # move the tag guys (but not too often)
        if self.time_total %2 ==0:
            self.TagList=self.MoveTag(self.TagList)
        """
        # move the tag guys (but give the agent a head start)
        if self.time_total >= 2:
            self.TagList=self.MoveTag(self.TagList)
        # don't let the enemies touch you
        
        for enemy in self.TagList:
            enemy_pos = np.array([enemy[0], enemy[1]])
            if np.array_equal(enemy_pos, self.agent_pos):
                reward -= 1000 # once you die get a reward for how long you lived
                terminated = True
                return (
                self._get_obs(),
                reward,
                terminated,
                False,
                self._get_info(),
                )
            else: #get points for being far away
                reward += abs(enemy_pos[1]-self.agent_pos[1])
                reward += abs(enemy_pos[0]-self.agent_pos[0])
        
       
        direction = self._action_to_direction[action]  
        truncated = False
        agent_new_pos = self.agent_pos + direction
        
        
        # want to encourage the agent to move to new tiles over visiting old ones 
        
        current_cell = tuple(agent_new_pos)
        
        #but really don't want the agent running into walls over and over again
            
        if self.visited.count(current_cell) >= 1:
            reward-=0
        else:
            self.visited.append(current_cell)
            reward+=0
            
        
        if (self.wall==agent_new_pos).all(1).any():
         #   reward-=1
            pass
        else:
            self.agent_pos = agent_new_pos
        
        
        if self.time_total >= 200:
                truncated = True
                reward += 1000 #big reward for winning
            
                
                
        observation = self._get_obs()
        info = self._get_info()
        reward = float(reward)  
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
        
    def render(self):
        # agent is represented as a cross, rest as a dot
        use_visited=np.array(self.visited)
        use_seen = np.array(self.seen)
        use_Tag = []
        
        for enemy in self.TagList:
            use_Tag.append([enemy[0],enemy[1]])
        use_Tag = np.array(use_Tag)
       
        if self.render_mode == "console":
            for y in range(self.height):
                for x in range(self.width):
                    if np.array_equal([x,y], self.agent_pos):
                        print(Fore.MAGENTA + "X", Style.RESET_ALL, end="")
                    elif np.any(np.all([x,y] == use_Tag, axis=1)):
                        print(Fore.RED + "2", Style.RESET_ALL, end="")
                    elif np.any(np.all([x,y] == self.wall, axis=1)):
                        if np.any(np.all([x,y] == use_visited, axis=1)):
                            print(Fore.BLUE+"1",Style.RESET_ALL, end="")
                        else:
                            print(Fore.CYAN+"1", Style.RESET_ALL, end="")
                    else:
                        if np.any(np.all([x,y] == use_visited, axis=1)):
                            print(Fore.BLUE+"0",Style.RESET_ALL, end="")
                        else:
                            print("0", end=" ")
                print("")
    def close(self):
        pass
    
    # make an enemy tag player
    def MakeTag(self, x,y, TagList):
        stepnumber =0
        direction = 0
        TagList.append([x,y])
        #position 
        return TagList
    
    def MoveTag(self, TagList):
    # chase the target
        for enemy in TagList:
            enemy_move_list = []
            enemy_pos=np.array(enemy)
            if np.array_equal(enemy_pos, self.agent_pos):
                return TagList
            
            if self.agent_pos[0] !=enemy[0]:
                enemy_move_list.append(np.array([(self.agent_pos[0]-enemy[0])/abs(self.agent_pos[0]-enemy[0]), 0]))
            if self.agent_pos[1] != enemy[1]:
                enemy_move_list.append(np.array([0, (self.agent_pos[1]-enemy[1])/abs(self.agent_pos[1]-enemy[1])]))
            
            direction = random.choice(enemy_move_list)
            enemy_new_pos=enemy_pos+direction
            while (self.wall == enemy_new_pos).all(1).any():
                direction = random.choice([np.array([1,0]),np.array([-1,0]),np.array([0,1]), np.array([0,-1])])
                enemy_new_pos=enemy_pos+direction
            enemy[0]=enemy_new_pos[0]
            enemy[1]=enemy_new_pos[1]
            
            
                
           
           
        return TagList
          
                
            
        

env=GoLeftEnv(width=20, height=20,enemy_width=2, enemy_height=18)
check_env(env, warn=True)
obs = env.reset()
env.render()

for step in range(50):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step + 1}")
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break
        
   



vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(width=20, height=20, enemy_width=2, enemy_height=18))
model = A2C("MlpPolicy", vec_env, verbose=1)

obs=vec_env.reset()
vec_env.render()
eval_env = vec_env
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                            deterministic=True, render=False)

model.learn(500000)

vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(width=20, height=20, enemy_width=18, enemy_height=18))
total_reward=np.array([])
obs = vec_env.reset()
n_steps = 500
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    total_reward = np.append(total_reward, reward)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "total reward=", np.sum(total_reward))
        break
