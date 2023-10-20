"""
Here is our skeleton object.

our agent obect should:
    have a position
    be able to move
    be able to look around
    be able to store what its seen
    be able to store where its been

"""


class Agent():
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y
        self._surroundings = []
    def set_xy(self, new_x, new_y):
        self._x = new_x
        self._y = new_y
    def look_around(self, new_environ):
        self._surroundings = new_environ
        
    def update_step(self):
        
        new_surr = look_around_weak(self)
        update_knowledge(new_surr)
        found_check()
        
    def look_around_weak(self, new_environ):
        """Look at the four surrouning cells"""
        new_surr=[]
        for idx, val in enumerate(self._surroundings):
            if val != 0:
                x_coord =  (idx-(idx%10)) / 10
                y_coord = idx%10
                
                if (((x_coord - self._x)**2) + ((y_coord - self._y)**2)) == 1:
                    new_surr.append(1)
                else:
                    new_surr.append(0)
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
    
        

agent_max = Agent(10,10)

my_environ = [1 if ((i > 30) & (i < 40)) else 0 for i in range(100) ]

agent_max.look_around(my_environ)

agent_max.set_xy(11,11)
print(agent_max.is_proximal())

agent_max.set_xy(5,5)
print(agent_max.is_proximal())
