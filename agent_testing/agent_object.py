"""
Here is our skeleton object.

our agent obect should:
    have a position
    be able to move
    be able to look around
    be able to store what its seen
    be able to store where its been

"""
import random
import time
from colorama import Fore, Back, Style
width = 30
height = 20

def make_enviro(width, height, content):
    enviro = []
    for i in range(height):
        height = []
        for j in range(width):
            height.append(content)
        enviro.append(height)
    return enviro
def print_enviro(environ):
    for i in environ:
        for j in i:
            print(j, end=" ")
        print("")

class Agent():
    def __init__(self, line_of_sight, x=0, y=0):
        self._x = x
        self._y = y
        self._surroundings = []
        self._room_knowledge = make_enviro(width, height, ".")
        self._line_of_sight = line_of_sight
    def set_xy(self, new_x, new_y):
        self._x = new_x
        self._y = new_y
    def look_around(self, new_environ):
        self._surroundings = new_environ
        
    def update_step(self, true_knowledge):
        self.look_around_weak(true_knowledge)
        
        for idy, value1 in enumerate(self._room_knowledge):
            for idx, value in enumerate(value1):
                if self._x == idx and self._y == idy:
                    print(Fore.MAGENTA + "X", Style.RESET_ALL, end="")
                else:
                    print(value, end=" ")
            print("")
    def look_around_weak(self, new_environ):
        """Look at the four surrouning cells"""
        x= self._x
        y=self._y
        if self._line_of_sight == False:
            if y != height-1:
                self._room_knowledge[y+1][x]=new_environ[y+1][x]
            if y !=0:
                self._room_knowledge[y-1][x]=new_environ[y-1][x]
            if x!= width-1:
                self._room_knowledge[y][x+1]=new_environ[y][x+1]
            if x!= 0:
                self._room_knowledge[y][x-1]=new_environ[y][x-1]
        if self._line_of_sight == "line":
            for i in range(y):
                self._room_knowledge[y-i][x]=new_environ[y-i][x]
                if new_environ[y-i][x] == 1:
                    break
            for i in range(height-y):
                self._room_knowledge[y+i][x]=new_environ[y+i][x]
                if new_environ[y+i][x] == 1:
                    break    
            for i in range(x):
                self._room_knowledge[y][x-i]=new_environ[y][x-i]
                if new_environ[y][x-i] == 1:
                    break
            for i in range(width-x):
                self._room_knowledge[y][x+i]=new_environ[y][x+i]
                if new_environ[y][x+i] == 1:
                    break 
                
    def is_proximal(self):
        """Look at surroundings. Return boolean about whether one is next to it. Assume 10x10 grid."""
        
        for idx, val in enumerate(self._surroundings):
            if val != 0:
                y_coord =  (idx-(idx%10)) / 10
                x_coord = idx%10
                
                if x_coord == self._x:
                    return True
                if y_coord == self._y:
                    return True
        
        return False
    
    def update_knowledge(self, new_surr):
        for i in range(100):
            if new_surr[i] !=".":
                self._room_knowledge[i] = new_surr[i]
                
    def move(self, direction, distance=1):
        if direction == 1:
            self.set_xy(self._x, self._y-distance)
        if direction == 2:
            self.set_xy(self._x, self._y+distance)
        if direction == 3:
            self.set_xy(self._x-distance, self._y)
        if direction == 4:
            self.set_xy(self._x+distance, self._y)
    """Brendan's possible knowledge search options """               
    def possible_moves(self, distance=1):
        """is a direction and distance ok """
        possible_moves = []
        y=self._y
        x=self._x
        if self._y >= distance and self._room_knowledge[y-distance][x] != 1:
            possible_moves.append(1)
        if self._y+distance < height and self._room_knowledge[y+distance][x] != 1:
            possible_moves.append(2)
        if self._x -distance >= 0 and self._room_knowledge[y][x-distance] != 1:
            possible_moves.append(3)    
        if self._x +distance <width and self._room_knowledge[y][x+distance] != 1:
            possible_moves.append(4)
        if len(possible_moves)>0:
            return possible_moves
        else:
            print("Father help I have fallen in an epistemic hole.")
            quit()
    def least_knowledge(self):
        """ Finds the direction we know the least about"""
        up_total = 0
        down_total = 0
        left_total = 0
        right_total = 0
        knowledge_ranking = []
        for idy, content in enumerate(self._room_knowledge):
            for idx, value in enumerate(content):
                if value == ".":
                    if idy <= self._y:
                        up_total += 1
                    if idy >= self._y:
                        down_total+=1
                    if idx <= self._x:
                        left_total += 1
                    if idx >= self._x:
                        right_total+=1
        l = [up_total, down_total, left_total, right_total]
        print(l)
        for i in range(4):
            m= max(l)
            if up_total == m:
                knowledge_ranking.append(1)
                l.remove(up_total)
          
            if down_total == m:
                knowledge_ranking.append(2)
                l.remove(down_total)
                
            if left_total == m:
                knowledge_ranking.append(3)
                l.remove(left_total)
                
            if right_total == m:
                knowledge_ranking.append(4)
                l.remove(right_total)
            if len(l)==0:
                break
        return knowledge_ranking
    def least_knowledge_move(self):
        knowledge_ranking = self.least_knowledge()
        possible = self.possible_moves()
        for i in knowledge_ranking:
            if i in possible:
                self.move(i)
                print("I am moving in direction", i) 
                return
        print("Father help I have fallen in an epistemic hole.")
        quit()
agent_max = Agent("line", 1,2)


my_environ_but_better = make_enviro(width,height, 0)
p = min(width, height)


for i in range(p):
    my_environ_but_better[i][i]=1
    

print_enviro(my_environ_but_better)
agent_max.update_step(my_environ_but_better)

for i in range(500):
    agent_max.least_knowledge_move()
    agent_max.update_step(my_environ_but_better)
    time.sleep(.5)

