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

class Agent():
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y
        self._surroundings = []
        self._room_knowledge = []
   
        for i in range(100):
            self._room_knowledge.append(".")
    def set_xy(self, new_x, new_y):
        self._x = new_x
        self._y = new_y
    def look_around(self, new_environ):
        self._surroundings = new_environ
        
    def update_step(self, true_knowledge):
        
        new_surr = self.look_around_weak(true_knowledge)
        self.update_knowledge(new_surr)
        for j in range(10):
            for i in range(10):
                if self._x+10*self._y ==j*10+i:
                    print("X", end=" ")
                else:
                    print(self._room_knowledge[j*10+i], end=" ")
            print("")
    def look_around_weak(self, new_environ):
        """Look at the four surrouning cells"""
        new_surr=[]
        for idx, val in enumerate(new_environ):
            y_coord =  (idx-(idx%10)) / 10
            x_coord = idx%10
            
            if (((x_coord - self._x)**2) + ((y_coord - self._y)**2)) == 1:
                if val == 1:
                    new_surr.append(1)
                else:
                    new_surr.append(0)
            else:
                new_surr.append(".")
        print(len(new_surr))
        return new_surr

        
    def is_proximal(self):
        """Look at surroundings. Return boolean about whether one is next to it. Assume 10x10 grid."""
        
        for idx, val in enumerate(self._surroundings):
            if val != 0:
                x_coord =  (idx-(idx%10)) / 10
                y_coord = idx%10
                
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
    def decide_move(self, distance=1):
        """check surroundings - build a list of possible actions - act """
        distance = 1
        position = self._x+10*self._y
        possible_moves = []
        if self._y >= distance and self._room_knowledge[position-10*distance] != 1:
            possible_moves.append(1)
        if self._y+distance < 10 and self._room_knowledge[position+10*distance] != 1:
            possible_moves.append(2)
        if self._x -distance >= 0 and self._room_knowledge[position-distance] != 1:
            possible_moves.append(3)    
        if self._x +distance <10 and self._room_knowledge[position+distance] != 1:
            possible_moves.append(4)
        if len(possible_moves)>0:
            return random.choice(possible_moves)
        else:
            print("Father help I have fallen in an epistemic hole.")
            quit()
                    
               
                    
                
              

agent_max = Agent(4,4)


my_environ = [1 if ((i > 30) & (i < 40)) else 0 for i in range(100) ]

agent_max.update_step(my_environ)

for i in range(50):
    move_choice = agent_max.decide_move()
    print(move_choice)
    agent_max.move(move_choice)
    agent_max.update_step(my_environ)
    time.sleep(.5)