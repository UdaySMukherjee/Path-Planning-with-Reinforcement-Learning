import numpy as np
import warnings
import matplotlib.pyplot as plt

thr = 10 # threshold distance to the terminal for making decision of the done flag
v = 10 
obstacle_width=10
# warnings.simplefilter("error")
warnings.simplefilter("ignore", UserWarning)

# Global variable for dictionary with coordinates for the final route
final_route = {}

class Environment(object):
  def __init__(self, initial_position, target_position,X_max, Y_max, num_actions):
    
    #Initial state of the system:
    self.state0 = np.zeros((2,11,11)) 
    self.state0[0][10][1] = 1 # robot initial position

    # self.Obstacle_x = [3,3,3,3,3,3,7,7,7,7,7,7]
    # self.Obstacle_y = [5,6,7,8,9,10,0,1,2,3,4,5]

    self.Obstacle_x = [4, 3, 3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 5, 5, 5, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9]
    self.Obstacle_y = [10, 4, 5, 6, 10, 6, 5, 4, 8, 9, 9, 8, 5, 6, 4, 0, 1, 2, 0, 1, 2, 8, 9, 10, 8, 9, 10]

    self.vector_obstacle_x=[0]*len(self.Obstacle_x)
    self.vector_obstacle_y=[0]*len(self.Obstacle_x)

    for i in range(len(self.Obstacle_x)):
      self.vector_obstacle_x[i]=10*(self.Obstacle_x[i]-0.5)
      self.vector_obstacle_y[i]=10*(10 - self.Obstacle_y[i] -0.5)
    
    self.obstacle =  [np.zeros((1, 4)).tolist() for i in range(len(self.Obstacle_x))]
    for i in range(len(self.vector_obstacle_x)):
      self.obstacle[i]=[self.vector_obstacle_x[i],self.vector_obstacle_y[i],obstacle_width,obstacle_width]

    for i in range(len(self.Obstacle_x)):
      self.state0[1, self.Obstacle_y[i], self.Obstacle_x[i]] = 1 

    self.state0[1][0][9] = 1 #the position of the Terminal
    self.X_max = X_max #range of X: X_max, the min is 0
    self.Y_max = Y_max #range of Y: Y_max, the min is 0
    self.vector_state0 = np.asarray(initial_position) #initial state，robot initial position, (10,10)
    self.Is_Terminal = False #achieve terminal or not, bool value, terminal = start point
    self.vector_agentState = np.copy(self.vector_state0) # state of the agent
    self.agentState = np.copy(self.state0) # state of the agent
   # self.visited_charger = 0 #visit the charge or not

    self.Terminal = np.asarray(target_position) #np.asarray([90., 90.]) # terminal 2
    self.doneType = 0 # flag showing type of done! 
    self.max_episode_steps = 10000 
    self.steps_counter = 0
    self.num_actions = num_actions #number of actions

  # Dictionaries to draw the final route
    self.dic = {}
    self.final_path = {}
    # Key for the dictionaries
    self.index = 0
    # Writing the final dictionary first time
    self.firstsuc= True
    # Showing the steps for longest found route
    self.longest = 0
    # Showing the steps for the shortest route
    self.shortest = 0

    self.actionspace = {0:[v,0], 1:[0,v], 2: [-v,0], 3: [0,-v], 4: [-v,v], \
                      5:[-v,-v], 6:[v,v], 7: [v,-v]} #8 actions
    # self.actionspace = {0:[v,0], 1:[0,v], 2: [-v,0], 3: [0,-v]} #action space, 4 actions
    
  def reset(self): 
    self.agentState = np.copy(self.state0)
    self.vector_agentState = np.copy(self.vector_state0)
    self.dic = {}
    self.index=0
    self.doneType = 0
    self.steps_counter = 0
    self.Is_Terminal = False
    return self.agentState

  def step(self, action): #agent interact with the environment through action
    V = self.actionspace[action]
    self.vector_agentState[0] += V[0] 
    self.vector_agentState[1] += V[1] 
    #if agent cross the boundary
    if self.vector_agentState[0] < 0:
      self.vector_agentState[0] = 0
    if self.vector_agentState[0] > 100:
      self.vector_agentState[0] = 100
    if self.vector_agentState[1] < 0:
      self.vector_agentState[1] = 0
    if self.vector_agentState[1] > 100:
      self.vector_agentState[1] = 100

    # Writing in the dictionary coordinates of found route
    self.dic[self.index] = self.vector_agentState.tolist() 

    # Updating key for the dictionary
    self.index += 1

    i_x = np.copy(self.vector_agentState[0])/10 
    i_y = 10 - np.copy(self.vector_agentState[1])/10 
    self.agentState = np.copy(self.state0) #2*11*11
    self.agentState[0][9][1] = 0
    self.agentState[0, int(i_y), int(i_x)] = 1 
    #self.energy_level -= self.propulsion_power(V) * T_s
    self.steps_counter +=1 #step accumulate 1
    self.Is_Terminal = self.isTerminal() # achieve the terminal or not
    
    reward,next_state_flag = self.get_reward(self.vector_agentState,action)     

    return self.agentState, next_state_flag,reward, self.Is_Terminal , None

  # function for judging whether agent achieve the terminal or not
  def isTerminal(self):

    Distance2Terminal = np.linalg.norm(np.subtract(self.vector_agentState , self.Terminal))

   # if d_.all() == True and Distance2Terminal**0.5 == 0 and self.energy_level > 0: ###self.agentState[2] > 0 :
    if Distance2Terminal**0.5 == 0: 
      self.doneType = 1
      return True
    else:
      return False

#function for geting rewards
  def get_reward(self,state,action):
#    ch, dist = self.channel()
    reward = 0 # initialize the reward as 0
    #Cooridinate change
    # i_x = int(np.copy(self.vector_agentState[0])/10)
    # i_y = int(10 - np.copy(self.vector_agentState[1])/10)
    
    # agent doesn't achieve the terminal
    if not self.Is_Terminal: 
       #judge whether the agent  crash the obstacle
      if self.is_collision(state):
          reward=-20
          next_state_flag = 'obstacle'
      else:
          if action==0 or action==1 or action==2 or action==3:
            reward=-1
          else:
            reward=-1.5
          next_state_flag = 'continue'

    elif self.doneType == 1:
        reward = 20
        next_state_flag = 'goal'
        # Filling the dictionary first time
        if self.firstsuc == True:
            for j in range(len(self.dic)):
                self.final_path[j] = self.dic[j]
            self.firstsuc = False
            self.longest = len(self.dic)
            self.shortest = len(self.dic)
      # Checking if the currently found route is shorter
        else:
          if len(self.dic) < len(self.final_path):
              # Saving the number of steps for the shortest route
              self.shortest = len(self.dic)
              # Clearing the dictionary for the final route
              self.final_path = {}
              # Reassigning the dictionary
              for j in range(len(self.dic)):
                  self.final_path[j] = self.dic[j] 

          # Saving the number of steps for the longest route
          if len(self.dic) > self.longest:
              self.longest = len(self.dic)
    return reward, next_state_flag 
  
  # Function to show the found route
  def final(self):
      # Showing the number of steps
      print('The shortest route:', self.shortest)
      print('The longest route:', self.longest)
      for j in range(len(self.final_path)):
      #     #Showing the coordinates of the final route
      #     #print(self.final_path[j])
        final_route[j] = self.final_path[j]

      # Plotting the environment
      plt.figure(figsize=(8, 8))
      plt.xlim(0, 100)
      plt.ylim(0, 100)

      # Plot obstacles
      for (x, y, w, h) in self.obstacle:
        plt.gca().add_patch(plt.Rectangle((x - 0.5 * w, y - 0.5 * h), w, h, color='gray'))

      # Plot the robot's final path
      x_vals = [self.final_path[j][0] for j in range(len(self.final_path))]
      y_vals = [self.final_path[j][1] for j in range(len(self.final_path))]
      plt.plot(x_vals, y_vals, '-o', label="Robot Path", color='blue')

      # Plot the starting point
      plt.scatter(self.vector_state0[0], self.vector_state0[1], color='green', s=100, label="Start")

      # Plot the terminal point
      plt.scatter(self.Terminal[0], self.Terminal[1], color='red', s=100, label="Terminal")

      # Labels and legend
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.title('Robot Path and Obstacles')
      plt.legend()

      # Save the plot as a PNG file
      plt.savefig("environment.png")
      print("Environment image saved as environment.png")

  def is_collision(self,state):
    delta = 0.5*obstacle_width
    for (x, y, w, h) in self.obstacle: 
      if 0 <= state[0] - (x - delta) <= w  \
            and 0 <= state[1] - (y - delta) <= h :
        return True

# Returning the final dictionary with route coordinates
# Then it will be used in agent.py
def final_states():
    return final_route
