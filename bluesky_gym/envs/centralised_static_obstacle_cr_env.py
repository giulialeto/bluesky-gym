import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import bluesky_gym.envs.common.deterministic_path_planning as path_plan
from bluesky.tools.aero import kts

import gymnasium as gym
from gymnasium import spaces

from shapely.geometry import Polygon

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1 # reach set waypoint

DRIFT_PENALTY = -0.01
AC_INTRUSION_PENALTY = -5
RESTRICTED_AREA_INTRUSION_PENALTY = -5

WAYPOINT_DISTANCE_MIN = 180 # KM
WAYPOINT_DISTANCE_MAX = 400 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

AC_DISTANCE_MIN = 50 # KM
AC_DISTANCE_MAX = 170 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts (check)

AC_SPD = 150 # kts
ALTITUDE = 350 # In FL

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10
## for obstacles generation
NUM_OBSTACLES = 10 #np.random.randint(1,5)
NUM_INTRUDERS = 5
NUM_AC = 10
NUM_AC_STATE = NUM_AC #check if this is not just redundant 
INTRUSION_DISTANCE = 5 # NM

## number of waypoints coincides with the number of destinations for each aircraft (actor and all other aircraft)
NUM_WAYPOINTS = NUM_AC

POLY_AREA_RANGE = (50, 1000) # In NM^2
# CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates
CENTER = (52., 4.) # TU Delft AE Faculty coordinates

MAX_DISTANCE = 350 # width of screen in km

class CentralisedStaticObstacleCREnv(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment
    The environment simulates a 2D airspace where an ownship (the agent) must navigate to a series of waypoints while avoiding collisions with other aircraft and static obstacles.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        # Observation space should include ownship and intruder info, end destination info for ownship, relative position of obstcles in reference to ownships
        # Maybe later also have an option for CNN based intruder and obstacle info, could be interesting
        self.observation_space = spaces.Dict(
            {   
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "intruder_y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "restricted_area_radius": spaces.Box(0, 1, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "restricted_area_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES, ), dtype=np.float64),
                "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2*NUM_AC_STATE,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        if bs.sim is None:
            bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')
        
        # variables for logging
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])
        self.average_v_action = [] # Why is this called AVERAGE? isn't it just the action?
        self.average_hdg_action = [] # Why is this called AVERAGE? isn't it just the action?
        self.ac_indices = np.arange(1, NUM_AC + 1)


        self.obstacle_names = []

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()
        self.counter = 0

        # reset logging variables 
        self.total_reward = 0
        self.waypoint_reached = 0
        self.crashed = 0
        self.average_drift = np.array([])
        self.impossible_route_counter = 0

        # bs.tools.areafilter.deleteArea(self.poly_name)

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(CENTER[0], CENTER[1], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]
        
        # generate obstacles and other aircraft until a valid (not more than two sectors overlap) scenario is found
        self.sample_obstacle = True
        while self.sample_obstacle:
            self._generate_obstacles()

        self._generate_aircraft()

        self._generate_waypoint()

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.counter += 1
        self._get_action(action)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
            reward, done, truncated = self._get_reward()
            if truncated:
                observation = self._get_obs()
                info = self._get_info()
                self.total_reward += reward
                return observation, reward, done, truncated, info

        observation = self._get_obs()
        reward, done, truncated = self._get_reward()
        self.total_reward += reward
        info = self._get_info()

        return observation, reward, done, truncated, info
    
    def _generate_aircraft(self, num_aircraft = NUM_AC):
        
        self.aircraft_names = []
        for i in range(num_aircraft): 
            aircraft_name = 'AC' + str(i+1)
            self.aircraft_names.append(aircraft_name)

            check_if_inside_obs = True
            loop_counter = 0
            # check if aircraft is is created inside obstacle
            while check_if_inside_obs:
                loop_counter+= 1

                aircraft_dis_from_reference = np.random.randint(AC_DISTANCE_MIN, AC_DISTANCE_MAX)
                aircraft_hdg_from_reference = np.random.randint(0, 360)
                
                aircraft_lat, aircraft_lon = fn.get_point_at_distance(CENTER[0], CENTER[1], aircraft_dis_from_reference, aircraft_hdg_from_reference)
                
                bs.traf.cre(acid=aircraft_name,actype="A320",aclat=aircraft_lat, aclon=aircraft_lon, acspd=AC_SPD)
                
                ac_idx = bs.traf.id2idx(aircraft_name)

                inside_temp = []
                for j in range(NUM_OBSTACLES):
                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], bs.traf.lat, bs.traf.lon, bs.traf.alt)[-1])

                check_if_inside_obs = any(x == True for x in inside_temp)
                if check_if_inside_obs:
                    bs.traf.delete(ac_idx)

                if loop_counter > 50:
                    raise Exception("No aircraft can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

    def _generate_polygon(self, centre):
        poly_area = np.random.randint(POLY_AREA_RANGE[0]*2, POLY_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        
        while p_area < POLY_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        p = [fn.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        
        return p_area, p, R
    
    def _generate_obstacles(self):
        # delete existing obstacles from previous episode in BlueSky
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)

        obstacle_names = []
        obstacle_vertices = []
        obstacle_radius = []

        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []

        self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        obstacle_dict = {}  # Initialize the dictionary to store obstacles for overlap checking

        for i in range(NUM_OBSTACLES):

            centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
            _, p, R = self._generate_polygon(centre_obst)
            
            points = [coord for point in p for coord in point] # Flatten the list of points
            poly_name = 'restricted_area_' + str(i+1)
            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            
            obstacle_names.append(poly_name)
            obstacle_vertices.append(obstacle_vertices_coordinates)
            obstacle_radius.append(R)

        for i in range(NUM_OBSTACLES):
            overlap_list = []  # List to store overlap information
            # Check for overlaps with existing obstacles
            for j in range(NUM_OBSTACLES):
                if i == j:
                    continue  # Skip checking the same obstacle

                found_overlap = False
                for k in range(0, len(obstacle_vertices[j])):
                    # check if the vertices of the obstacle are inside the other obstacles
                    overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array(obstacle_vertices[j][k][0]), np.array(obstacle_vertices[j][k][1]), np.array([ALTITUDE]))[0]
                    if overlap:
                        overlap_list.append(obstacle_names[j])
                        break #break vertex loop
                    # check if points along the edges of the obstacle are inside the other obstacles
                    if k == len(obstacle_vertices[j]) -1:
                        interpolated_points = self._interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][0])
                    else:
                        interpolated_points = self._interpolate_along_obstacle_vertices(obstacle_vertices[j][k], obstacle_vertices[j][k+1])
                    for point in interpolated_points:
                        overlap = bs.tools.areafilter.checkInside(obstacle_names[i], np.array(point[0]), np.array(point[1]), np.array([ALTITUDE]))[0]
                        if overlap:
                            overlap_list.append(obstacle_names[j])
                            found_overlap = True
                            break # break interpolation loop
                    if found_overlap:
                        break
        
            obstacle_dict[obstacle_names[i]] = overlap_list
        
        max_overlaps_allowed = 1

        too_many_overlapping_obstacles = any(len(overlaps) > max_overlaps_allowed for overlaps in obstacle_dict.values())

        if too_many_overlapping_obstacles:
            self.sample_obstacle = True
        else:
            self.sample_obstacle = False

        # Store the generated obstacles in the environment
        self.obstacle_names = obstacle_names
        self.obstacle_vertices = obstacle_vertices
        self.obstacle_radius = obstacle_radius

    def _interpolate_along_obstacle_vertices(self, vertex_1, vertex_2, n=15):
        """Interpolate n points between vertex_1 and vertex_2."""
        lats = np.linspace(vertex_1[0], vertex_2[0], n)
        lons = np.linspace(vertex_1[1], vertex_2[1], n)
        return list(zip(lats, lons))

    def _generate_waypoint(self):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            check_inside_var = True
            loop_counter = 0
            while check_inside_var:
                loop_counter += 1
                # if i == 0:
                #     wpt_dis_init = np.random.randint(100, 170)
                # else:
                #     wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
                wpt_dis_init = np.random.randint(100, 170)
                wpt_hdg_init = np.random.randint(0, 360)
                wpt_lat, wpt_lon = fn.get_point_at_distance(CENTER[0], CENTER[1], wpt_dis_init, wpt_hdg_init)

                # working around the bug in bluesky that gives a ValueError when checkInside is used to check a single element
                wpt_lat_array = np.array([wpt_lat, wpt_lat])
                wpt_lon_array = np.array([wpt_lon, wpt_lon])
                ac_idx_alt_array = np.array([ALTITUDE, ALTITUDE])
                inside_temp = []
                for j in range(NUM_OBSTACLES):
                    # shapetemp = bs.tools.areafilter.basic_shapes[self.obstacle_names[j]]
                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], wpt_lat_array, wpt_lon_array, ac_idx_alt_array)[0])

                check_inside_var = any(x == True for x in inside_temp)                    
                
                if loop_counter > 1000:
                    raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)
            
            ac_idx_aircraft = bs.traf.id2idx(self.aircraft_names[i])
            initial_wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx_aircraft], bs.traf.lon[ac_idx_aircraft], self.wpt_lat[i], self.wpt_lon[i])
            bs.traf.hdg[ac_idx_aircraft] = initial_wpt_qdr
            bs.traf.ap.trk[ac_idx_aircraft] = initial_wpt_qdr

    def _generate_coordinates_centre_obstacles(self, num_obstacles = NUM_OBSTACLES):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        
        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)

            obstacle_centre_lat, obstacle_centre_lon = fn.get_point_at_distance(CENTER[0], CENTER[1], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)
    
    def _get_obs(self):
        ac_idx = bs.traf.id2idx('AC1')

        self.intruder_distance = []
        self.intruder_cos_bearing = []
        self.intruder_sin_bearing = []
        self.intruder_x_difference_speed = []
        self.intruder_y_difference_speed = []

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        # obstacles 
        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []
        
        self.ac_hdg = bs.traf.hdg[ac_idx]
        
        # other aircraft destination waypoints
        self.other_ac_destination_waypoint_distance = []

        # intruders observation
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.intruder_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.intruder_sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.intruder_x_difference_speed.append(x_dif)
            self.intruder_y_difference_speed.append(y_dif)


        # destination waypoint observation
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
    
        self.destination_waypoint_distance.append(wpt_dis * NM2KM)
        self.wpt_qdr.append(wpt_qdr)

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))

        # other aircraft destination waypoints
        for i in range(NUM_INTRUDERS):
            other_ac_idx = i+1
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[other_ac_idx], bs.traf.lon[other_ac_idx], self.wpt_lat[other_ac_idx], self.wpt_lon[other_ac_idx])
            self.other_ac_destination_waypoint_distance.append(wpt_dis * NM2KM)

        # obstacles observations
        for obs_idx in range(NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        observation = {
                "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "intruder_cos_difference_pos": np.array(self.intruder_cos_bearing),
                "intruder_sin_difference_pos": np.array(self.intruder_sin_bearing),
                "intruder_x_difference_speed": np.array(self.intruder_x_difference_speed)/AC_SPD,
                "intruder_y_difference_speed": np.array(self.intruder_y_difference_speed)/AC_SPD,
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius)/(POLY_AREA_RANGE[0]),
                "restricted_area_distance": np.array(self.obstacle_centre_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(self.obstacle_centre_cos_bearing),
                "sin_difference_restricted_area_pos": np.array(self.obstacle_centre_sin_bearing),
            }

        return observation
    
    def _get_info(self):
        # Any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        return {
            'total_reward': self.total_reward,
            'waypoint_reached': self.waypoint_reached,
            'crashed': self.crashed,
            'average_drift': self.average_drift.mean(),
            'average_vinput': np.mean(self.average_v_action),
            'average_hdginput': np.mean(self.average_hdg_action)

        }

    def _get_reward(self):
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_other_ac_reward = self._check_intrusion_other_ac()
        intrusion_reward, intrusion_terminate = self._check_intrusion()

        total_reward = reach_reward + drift_reward + intrusion_other_ac_reward + intrusion_reward

        done = 0
        if self.wpt_reach[0] == 1:
            done = 1
        elif intrusion_terminate:
            done = 1

        return total_reward, done, False
    
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.destination_waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.waypoint_reached = 1
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        
        # for other_ac_wpt_reach_idx in range(len(self.wpt_reach)-1):
        #     if self.other_ac_destination_waypoint_distance[other_ac_wpt_reach_idx] < DISTANCE_MARGIN and self.wpt_reach[other_ac_wpt_reach_idx+1] != 1:
        #         self.wpt_reach[other_ac_wpt_reach_idx+1] = 1

        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.destination_waypoint_drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY
    
    def _check_intrusion_other_ac(self):
        ac_idx = bs.traf.id2idx('AC1')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                reward += AC_INTRUSION_PENALTY
        return reward

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('AC1')
        reward = 0
        terminate = 0
        for obs_idx in range(NUM_OBSTACLES):
            if bs.tools.areafilter.checkInside(self.obstacle_names[obs_idx], np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])):
                reward += RESTRICTED_AREA_INTRUSION_PENALTY
                self.crashed = 1
                terminate = 1
        return reward, terminate

    def _get_action(self,action):
        # dh = action[0] * D_HEADING
        # dv = action[1] * D_SPEED
        # heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('AC1')] + dh)
        # speed_new = (bs.traf.tas[bs.traf.id2idx('AC1')] + dv) * MpS2Kt

        # bs.stack.stack(f"HDG {'AC1'} {heading_new}")
        # bs.stack.stack(f"SPD {'AC1'} {speed_new}")


        for i in range(len(self.ac_indices)): # TODO: change to controled. now is but might be wrong..
            action_index = i
            # if not self.wpt_reach:
            # dh = action[action_index] * D_HEADING
            # else:
            #     dh = -self.drift
            # import debug
            # debug.red(action_index*2)
            # debug.green(action_index*2+1)
            #check with Sasha I think there is a bug here... +1 should be *2 +1
            # dv = action[action_index+1] * D_SPEED
            # dh = action[action_index] * D_HEADING

            dv = action[action_index*2+1] * D_SPEED
            dh = action[action_index*2] * D_HEADING
            self.average_v_action.append(dv)
            self.average_hdg_action.append(dh)

            ind_ac = self.ac_indices[i] # this is to map the controll-ee to the instantaneous action space (since we change ac)

            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(f'AC{ind_ac}')] + dh)
            speed_new = (bs.traf.tas[bs.traf.id2idx(f'AC{ind_ac}')] + dv) * MpS2Kt
            # speed_new = speed_new if speed_new>0 else 0
            
            # if not baseline_test:
            bs.stack.stack(f"HDG AC{ind_ac} {heading_new}")
            bs.stack.stack(f"SPD AC{ind_ac} {speed_new}")


    def _render_frame(self):
        # options for rendering
        hide_other_target_waypoints = True

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen_coords = self.screen_coords

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        px_per_km = self.window_width/MAX_DISTANCE

        # draw ownship



        # draw intruders
        ac_length = 3

        for i in range(NUM_AC):
            hdg = bs.traf.hdg[i]
            heading_end_x = ((np.sin(np.deg2rad(hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
            heading_end_y = ((np.cos(np.deg2rad(hdg)) * ac_length)/MAX_DISTANCE)*self.window_width

            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[i], bs.traf.lon[i])

            # determine color
            # if dis < INTRUSION_DISTANCE:
            #     color = (220,20,60)
            # else: 
            #     color = (80,80,80)
            # color = (235, 52, 52)#(220,20,60)
            color = (80,80,80)
            color_circle = (80,80,80)

            # check if the aircraft crashes onto another aircraft
            for other_ac_idx in range(NUM_AC):
                if other_ac_idx == i:
                    continue
                _, ac_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], bs.traf.lat[other_ac_idx], bs.traf.lon[other_ac_idx])
                if ac_dis < INTRUSION_DISTANCE:
                    color = (220,20,60)
                    color_circle = (220,20,60)
                    break
                    

            x_pos = (np.sin(np.deg2rad(qdr))*(dis * NM2KM)/MAX_DISTANCE)*self.window_width
            y_pos = -(np.cos(np.deg2rad(qdr))*(dis * NM2KM)/MAX_DISTANCE)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
            )

            # draw heading line
            heading_length = 10
            heading_end_x = ((np.sin(np.deg2rad(hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
            heading_end_y = ((np.cos(np.deg2rad(hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
            
            color_heading = (0,0,0)
            
            pygame.draw.line(canvas,
                color_heading,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )
            
            pygame.draw.circle(
                canvas, 
                color_circle,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/MAX_DISTANCE)*self.window_width,
                width = 2
            )


        # draw obstacles
        for vertices in self.obstacle_vertices:
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis*NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                points.append((x_ref, y_ref))
            pygame.draw.polygon(canvas,
                (0,0,0), points
            )

        # draw target waypoint
        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            
            indx += 1
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width


            if reach:
                color = (5, 128, 9)
            else:
                color = (235, 52, 52)

            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = (DISTANCE_MARGIN/MAX_DISTANCE)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        
        pygame.display.update()
        
        self.clock.tick(self.metadata["render_fps"])
        # pygame.time.wait(10**5)

        if self.counter == 1:
            pygame.time.wait(100)

    def close(self):
        bs.stack.stack('quit')
