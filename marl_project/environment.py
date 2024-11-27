import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiAgentShipTowEnv(gym.Env):
    def __init__(self,
                grid_size=300,
                dock_position=(50, 250),
                dock_dim=(200, 50),
                target_position=(50, 200),
                target_dim=(200, 50),
                frame_update=0.01,
                ship_dim=(60, 8),
                ship_mass=0.1,
                ship_inertia=0.1,
                ship_velocity=0.75,
                ship_angular_velocity=0.001,
                tugboat_dim=(5, 2),
                max_rope_length=10.0,
                linear_drag_coeff=0.5,
                angular_drag_coeff=4.0
                ):
        """
        Initialize the environment.
        """
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.dock_position = dock_position
        self.dock_dim = dock_dim
        self.target_position = target_position
        self.target_dim = target_dim
        self.dt = frame_update
        
        # Ship parameters
        self.ship_dim = ship_dim
        self.ship_mass = ship_mass
        self.ship_inertia = ship_inertia
        self.ship_velocity = np.array([ship_velocity, ship_velocity])
        self.angular_velocity = ship_angular_velocity
        
        # Tugboat parameters
        self.tugboat_dim = tugboat_dim
        self.max_rope_length = max_rope_length
        self.front_offset = ship_dim[0]
        self.z = np.sqrt((self.ship_dim[0]**2) + (self.ship_dim[1]**2))
        
        # Water drag
        self.linear_drag_coeff = linear_drag_coeff
        self.angular_drag_coeff = angular_drag_coeff

        # for each agent (tugboat)
        self.action_space = {
            'tugboat_1': spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([50.0, 50.0]),
                dtype=np.float32
            ),
            'tugboat_2': spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([50.0, 50.0]),
                dtype=np.float32
            )
        }


        self.observation_space = {
            'tugboat_1': spaces.Box(
                low=np.array([
                        0.0,                    # min x position (ship)
                        0.0,                    # min y position (ship)
                        -np.pi,                 # min rotation (ship)
                        0.0,                    # min x position (own tugboat)
                        0.0,                    # min y position (own tugboat)
                        -np.pi,                 # min rotation (own tugboat)
                        0.0,                    # min x position (other tugboat)
                        0.0,                    # min y position (other tugboat)
                        -np.pi,                  # min rotation (other tugboat)
                        0.5,                    # min distance to target
                        0.0                     # min rope length
                            ], dtype=np.float32),
                high=np.array([
                        self.grid_size,         # max x position (ship)
                        self.grid_size,         # max y position (ship)
                        np.pi,                  # max rotation (ship)
                        self.grid_size,         # max x position (own tugboat)
                        self.grid_size,         # max y position (own tugboat)
                        np.pi,                  # max rotation (own tugboat)
                        self.grid_size,         # max x position (other tugboat)
                        self.grid_size,         # max y position (other tugboat)
                        np.pi,                  # max rotation (other tugboat)
                        np.sqrt(2)*self.grid_size, # max possible distance (diagonal)
                        self.max_rope_length    # max rope length
                            ], dtype=np.float32)),
            'tugboat_2': spaces.Box(
                low=np.array([
                        0.0,                    # min x position (ship)
                        0.0,                    # min y position (ship)
                        -np.pi,                 # min rotation (ship)
                        0.0,                    # min x position (own tugboat)
                        0.0,                    # min y position (own tugboat)
                        -np.pi,                 # min rotation (own tugboat)
                        0.0,                    # min x position (other tugboat)
                        0.0,                    # min y position (other tugboat)
                        -np.pi,                 # min rotation (other tugboat)
                        0.5,                    # min distance to target
                        0.0                     # min rope length
                            ], dtype=np.float32),
                high=np.array([
                        self.grid_size,         # max x position (ship)
                        self.grid_size,         # max y position (ship)
                        np.pi,                  # max rotation (ship)
                        self.grid_size,         # max x position (own tugboat)
                        self.grid_size,         # max y position (own tugboat)
                        np.pi,                  # max rotation (own tugboat)
                        self.grid_size,         # max x position (other tugboat)
                        self.grid_size,         # max y position (other tugboat)
                        np.pi,                  # max rotation (other tugboat)
                        np.sqrt(2)*self.grid_size, # max possible distance (diagonal)
                        self.max_rope_length    # max rope length
                            ], dtype=np.float32)
            )
        }

        self.bound = 25
        self.obstacles = [
                          (0, 0, self.grid_size, self.bound),
                          (self.grid_size-self.bound, 0, self.bound, self.grid_size),
                          (0, 0, self.bound, self.grid_size),
                          (0, self.grid_size-self.bound, self.grid_size, self.bound),
                          (0, 125, 150, 25)
                        #   (300, 125, 100, 25)
                          ]  # (x, y, width, height)
        self.target_block = [self.target_position[0], self.target_position[1], self.target_dim[0], self.target_dim[1]]
        


    def reset(self) -> dict:
        """
        Reset the environment.

        Returns:
            observations (dict): Observations for each agent.
        """
        ship_state = np.array([
            50.0, 50.0, 0.0,  # Ship position and orientation
        ])
        
        tugboat1_state = np.array([
            55.0 + self.front_offset, 50.0, 0.0,  # pos and orientation
        ])
        
        tugboat2_state = np.array([
            55.0 + self.front_offset, 50.0 + self.ship_dim[1], 0.0,  # pos and orientation
        ])
        
        self.state = np.concatenate([
            ship_state,
            tugboat1_state,
            tugboat2_state,
            [np.sqrt((ship_state[0] - self.target_position[0])**2 + (ship_state[1] - self.target_position[1])**2)],  # Distance to target
            [self.max_rope_length]
        ])
        
        return self._get_observations()
    
    

    def _get_observations(self) -> dict:
        """
        Get observations for each agent.

        Returns:
            observations (dict): Observations for each agent.
        """
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state
        
        obs_tugboat1 = np.array([
            xs, ys, thetas,          # Ship state
            xt1, yt1, thetat1,       # Own state
            xt2, yt2, thetat2,         # Other tugboat state
            ds,                   # Distance to target
            np.linalg.norm(np.array([xt1, yt1]) - np.array([xs, ys]))  # rope length
        ])
        
        obs_tugboat2 = np.array([
            xs, ys, thetas,          # Ship state
            xt2, yt2, thetat2,       # Own state
            xt1, yt1, thetat1,         # Other tugboat state
            ds,                   # Distance to target
            np.linalg.norm(np.array([xt2, yt2]) - np.array([xs, ys]))  # rope length
        ])
        
        return {
            'tugboat_1': obs_tugboat1,
            'tugboat_2': obs_tugboat2
        }
    

    def extract_corners(self, x, y, l, b, theta) -> tuple:
        """
        Extract corners of the object.

        Args:
            x (float): x position of the object.
            y (float): y position of the object.
            l (float): length of the object.
            b (float): breadth of the object.
            theta (float): orientation of the object.
        
        Returns:
            obj_d (float): diagonal length of the object.
            obj_corner (list): list of corners of the object.
        """
        obj_d = np.sqrt(l**2 + b**2)
        obj_corner = [
            (x, y),
            (x + l*np.cos(theta), y + l*np.sin(theta)),
            (x + obj_d*np.cos((np.arctan(b/l)) + theta), y + obj_d*np.sin((np.arctan(b/l)) + theta)),
            (x - b*np.cos(np.pi/2 - theta), y + b*np.sin(np.pi/2 - theta))
        ]

        return obj_d, obj_corner

    
    def check_collision(self, x, y, l, b, theta) -> bool:
        """
        Check collision of the object with obstacles.

        Args:
            x (float): x position of the object.
            y (float): y position of the object.
            l (float): length of the object.
            b (float): breadth of the object.
            theta (float): orientation of the object.

        Returns:
            collision (bool): True if collision occurs, False otherwise.
        """
        obj_d, obj_corner = self.extract_corners(x, y, l, b, theta)

        for obstacle in self.obstacles:
            ox, oy, ow, oh = obstacle

            for corner in obj_corner:
                if (corner[0] > ox and corner[0] < (ox + ow)) and (corner[1] > oy and corner[1] < (oy + oh)):
                    return True
                else:
                    continue
        return False



    def calculate_distance_to_obstacle(self, x, y, l, b, theta) -> float:
        """
        Calculate distance to obstacle.

        Args:
            x (float): x position of the object.
            y (float): y position of the object.
            l (float): length of the object.
            b (float): breadth of the object.
            theta (float): orientation of the object.

        Returns:
            min_distance (float): minimum distance to obstacle.
        """
        min_distance = float('inf')
        
        obj_d, obj_corner = self.extract_corners(x, y, l, b, theta)
        
        for obstacle in self.obstacles:
            ox, oy, ow, oh = obstacle
            
            # obstacle corners
            obs_corners = [
                (ox, oy),
                (ox + ow, oy), 
                (ox, oy + oh), 
                (ox + ow, oy + oh)  
            ]

            def find_distance(obj_midpoint, x, y):
                return np.sqrt((obj_midpoint[0] - x)**2 + (obj_midpoint[1] - y)**2)
            
            for corner in obj_corner:
                if (corner[0] > ox and corner[0] < (ox + ow)) and (corner[1] > oy and corner[1] < (oy + oh)):
                    return 0.0
                
            min_list = []
            obj_midpoint = (((obj_corner[0][0] + obj_corner[2][0])/2), ((obj_corner[0][1] + obj_corner[2][1])/2))
            for corner in obs_corners:
                dist = find_distance(obj_midpoint, corner[0], corner[1])
                min_list.append(dist)
            min_distance = min(min_list)

        return min_distance
    

    def inside_target(self, x, y, theta) -> bool:
        """
        Check if the object is inside the target.

        Args:
            x (float): x position of the object.
            y (float): y position of the object.
            theta (float): orientation of the object.

        Returns:
            inside (bool): True if object is inside the target, False otherwise.
        """
        tx, ty, l, b = self.target_block

        obj_d, obj_corner = self.extract_corners(x, y, self.ship_dim[0], self.ship_dim[1], theta)
        c = 0
        for corner in obj_corner:
            if (corner[0] > tx and corner[0] < (tx + l)) and (corner[1] > ty and corner[1] < (ty + b)):
                c += 1
        if c == 4:
            return True
        else:
            return False


    
    def calculate_proximity_penalty(self, distance, danger_zone=25.0, max_penalty=50.0) -> float:
        """
        Calculate proximity penalty.

        Args:
            distance (float): distance to obstacle.
            danger_zone (float): danger zone radius.
            max_penalty (float): maximum penalty.

        Returns:
            penalty (float): proximity penalty.
        """
        if distance >= danger_zone:
            return 0.0
        
        penalty_factor = abs(2*(danger_zone - distance) / danger_zone)
        return max_penalty * penalty_factor


    def _get_reward(self, agent_id, new_ds) -> float:
        """
        Get reward for the agent.

        Args:
            agent_id (str): agent id.
            new_ds (float): new distance to target.

        Returns:
            reward (float): reward for the agent.
        """
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state
        
        # Base reward based on progress toward target
        reward = -ds / (self.grid_size*0.004)
        
        # Calculate proximity penalties
        ship_distance = self.calculate_distance_to_obstacle(
            xs, ys, self.ship_dim[0], self.ship_dim[1], thetas
        )/2
        ship_penalty = self.calculate_proximity_penalty(ship_distance)
        
        # Calculate tugboat penalties
        if agent_id == 'tugboat_1':
            tug_pos = (xt1, yt1, thetat1)
        else:
            tug_pos = (xt2, yt2, thetat2)
            
        tug_distance = self.calculate_distance_to_obstacle(
            tug_pos[0], tug_pos[1], self.tugboat_dim[0], self.tugboat_dim[1], tug_pos[2]
        )/2
        tug_penalty = self.calculate_proximity_penalty(tug_distance)
        
        # Apply penalties
        reward -= (ship_penalty + tug_penalty)
        
        # Penalty for stretching rope too much
        if agent_id == 'tugboat_1':
            rope_length = np.linalg.norm(np.array([xt1, yt1]) - np.array([xs, ys]))
        else:
            rope_length = np.linalg.norm(np.array([xt2, yt2]) - np.array([xs, ys]))
            
        if rope_length > self.max_rope_length:
            reward -= (rope_length - self.max_rope_length)*0.5
        
        # if new_ds < ds:
        #     reward += 10
        # Success reward
        # if ds < 6.0 and abs(thetas) < 0.1:
        #     reward += 1000.0
        if self.inside_target(xs, ys, thetas):
            reward += 1000
            
        # Immediate collision penalty
        # if self.check_collision(xs, ys, self.ship_dim[0], self.ship_dim[1], 'ship'): # replace agent_id with ship
        #     reward -= 100.0
        if tug_distance < 1:
            reward -= 100
            
        return reward
    


    def step(self, actions) -> tuple:
        """
        Step the environment.

        Args:
            actions (dict): Actions for each agent.

        Returns:
            observations (dict): Observations for each agent.
            rewards (dict): Rewards for each agent.
            dones (dict): Done flags for each agent.
            {} (dict): Additional information.
        """
        vx1, vy1 = actions['tugboat_1']
        vx2, vy2 = actions['tugboat_2']
        
       
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state

        # Attachment points
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                        ys + self.front_offset * np.sin(thetas)])
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                        ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        xt1 += vx1 * self.dt
        yt1 += vy1 * self.dt
        xt2 += vx2 * self.dt
        yt2 += vy2 * self.dt

        rope1_vector = np.array([xt1, yt1]) - P_fr
        rope2_vector = np.array([xt2, yt2]) - P_fl

        rope1_length = np.linalg.norm(rope1_vector)
        rope2_length = np.linalg.norm(rope2_vector)

        # Limit rope length
        if rope1_length > self.max_rope_length:
            xt1, yt1 = P_fr + (rope1_vector / rope1_length) * self.max_rope_length

        if rope2_length > self.max_rope_length:
            xt2, yt2 = P_fl + (rope2_vector / rope2_length) * self.max_rope_length

        force_front_1 = (rope1_vector / self.max_rope_length) * 2.0
        force_front_2 = (rope2_vector / self.max_rope_length) * 2.0

        net_force = force_front_1 + force_front_2
        linear_drag_force = -self.linear_drag_coeff * self.ship_velocity
        net_force += linear_drag_force

        # Update ship position
        acceleration = net_force / self.ship_mass
        self.ship_velocity += acceleration * self.dt
        xs += self.ship_velocity[0] * self.dt
        ys += self.ship_velocity[1] * self.dt

        # Update ship orientation
        torque = np.cross(P_fr - np.array([xs, ys]), force_front_1 + force_front_2)
        angular_drag_torque = -self.angular_drag_coeff * self.angular_velocity
        torque += angular_drag_torque
        angular_acceleration = torque / self.ship_inertia
        self.angular_velocity += angular_acceleration * self.dt
        thetas += self.angular_velocity * self.dt

        # Update distance to target
        new_ds = np.linalg.norm(np.array([xs+self.front_offset, ys+self.front_offset]) - np.array(self.target_position))

        # Update state
        self.state = np.array([xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, new_ds, l])

        # Get observations and rewards
        observations = self._get_observations()
        rewards = {
            'tugboat_1': self._get_reward('tugboat_1', new_ds),
            'tugboat_2': self._get_reward('tugboat_2', new_ds)
        }

        done = False
        
        # Check termination
        done_tugboat_1 = self.check_collision(xt1, yt1, 
                                              self.tugboat_dim[0], self.tugboat_dim[1], 
                                              thetat1)
        done_tugboat_2 = self.check_collision(xt2, yt2, 
                                              self.tugboat_dim[0], self.tugboat_dim[1], 
                                              thetat2)
        done_ship = self.inside_target(xs, ys, thetas) or self.check_collision(xs, ys, 
                                                                       self.ship_dim[0]*np.cos(thetas), 
                                                                       self.ship_dim[1]*np.sin(thetas),
                                                                       thetas)
        if done_ship or done_tugboat_1 or done_tugboat_2:
            done = True
        dones = {
            'tugboat_1': done,
            'tugboat_2': done,
            '__all__': done
        }

        return observations, rewards, dones, {}



    def render(self):
        """
        Render the environment.
        """
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()

        self.ax.set_facecolor('lightblue')
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xticks(np.arange(0, self.grid_size+1, 50))  # 10 unit spacing for X-axis
        self.ax.set_yticks(np.arange(0, self.grid_size+1, 50))  # 10 unit spacing for Y-axis
        self.ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray')

        # Draw dock
        dock_patch = patches.Rectangle(
            xy=self.dock_position,
            width=self.dock_dim[0],
            height=self.dock_dim[1],
            edgecolor='brown',
            facecolor='sienna'
        )
        self.ax.add_patch(dock_patch)

        # Unpack state
        xs, ys, thetas, xt1, yt1, thetat1, xt2, yt2, thetat2, ds, l = self.state

        # Draw ship
        ship_patch = patches.Rectangle(
            # (xs - self.ship_dim[0], ys - self.ship_dim[1]),
            (xs, ys),
            self.ship_dim[0],
            self.ship_dim[1],
            angle=np.degrees(thetas),
            edgecolor='grey',
            facecolor='grey'
        )
        self.ax.add_patch(ship_patch)

        # Draw tugboats
        tugboat1_patch = patches.Rectangle(
            (xt1, yt1),
            self.tugboat_dim[0],
            self.tugboat_dim[1],
            angle=np.degrees(thetas),
            edgecolor='green',
            facecolor='green'
        )
        tugboat2_patch = patches.Rectangle(
            (xt2 , yt2),
            self.tugboat_dim[0],
            self.tugboat_dim[1],
            angle=np.degrees(thetas),
            edgecolor='orange',
            facecolor='yellow'
        )
        self.ax.add_patch(tugboat1_patch)
        self.ax.add_patch(tugboat2_patch)

        # Draw ropes
        P_fr = np.array([xs + self.front_offset * np.cos(thetas),
                        ys + self.front_offset * np.sin(thetas)])
        P_fl = np.array([xs + self.z*np.cos(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0])),
                        ys + self.z*np.sin(thetas+np.arctan(self.ship_dim[1]/self.ship_dim[0]))])

        self.ax.plot([P_fr[0], xt1], [P_fr[1], yt1], 'k-', lw=1)
        self.ax.plot([P_fl[0], xt2], [P_fl[1], yt2], 'k-', lw=1)

        # Draw obstacles
        for (ox, oy, owidth, oheight) in self.obstacles:
            obs_patch = patches.Rectangle(
                (ox, oy),
                width=owidth,
                height=oheight,
                edgecolor='red',
                facecolor='lightcoral'
            )
            self.ax.add_patch(obs_patch)

        plt.pause(0.001)
        plt.draw()

if __name__ == "__main__":
    # Create environment
    env = MultiAgentShipTowEnv()
    
    observations = env.reset()
    done = False

    target_position = env.target_position

    while not done:
        actions = {
            'tugboat_1': env.action_space['tugboat_1'].sample(),
            'tugboat_2': env.action_space['tugboat_2'].sample()
        }

        observations, rewards, dones, _ = env.step(actions)
        print(f"Rewards: {rewards}")
        ship_x = observations['tugboat_1'][0]  

        if dones['__all__']:
            done = True
        env.render()

    env.close()
    plt.close('all')