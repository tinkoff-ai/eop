#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from dataclasses           import dataclass
from collections           import defaultdict
from heapq                 import heappop, heappush
from typing                import List
from itertools             import count

class FourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, 
        dense_reward=False, expert_eps=0.0, solutions_path=None):
        # Path to the solutions
        self._path_to_solutions = solutions_path
        # Since wall holes generated randomly, we need to rebuild an expert path every time
        # (because we use caching inside)
        self._expert_solutions = FourRoomsSolutions(env=self, path=self._path_to_solutions)

        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self._expert_eps       = expert_eps
        super().__init__(grid_size=19, max_steps=100)

        # Redefine action space
        self.action_space = spaces.Discrete(3)

        # Dense reward + Expert solutions
        self._dense_reward     = dense_reward
        if self._dense_reward:
            self.reward_range = (-1, 1)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # Count number of wall
        # Box walls + Intermediate Walls - 4 entrances - 5 intersecting tiles
        self._num_walls = (width * 2 + height * 2) - 4 - 1

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Place a goal
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
            self._goal_pos = self._goal_default_pos
        else:
            self._goal_pos = self.place_obj(Goal())

        # Place an agent
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        self.mission = 'Reach the goal'

    def _reward(self):
        """
        Compute the reward to be given upon success (and not only)
        """
        if self._dense_reward:
            prev_distance = len(self._expert_solutions.get_solution_at(self.prev_agent_pos, self.prev_agent_dir, self._goal_pos).actions)
            cur_distance  = len(self._expert_solutions.get_solution_at(self.agent_pos, self.agent_dir, self._goal_pos).actions)
            return prev_distance - cur_distance
        else:
            return 1 - 0.9 * (self.step_count / self.max_steps)

    def step(self, action):
        # Save previous position and direction
        self.prev_agent_pos = self.agent_pos
        self.prev_agent_dir = self.agent_dir

        obs, reward, done, info = MiniGridEnv.step(self, action)

        # Make reward dense if specified
        if self._dense_reward:
            reward = self._reward()

        return obs, reward, done, info

    def reset(self):
        obs = MiniGridEnv.reset(self)

        # Keep vars for expert solution
        self.initial_agent_pos = self.agent_pos
        self.initial_agent_dir = self.agent_dir

        self.prev_agent_pos = self.agent_pos
        self.prev_agent_dir = self.agent_sees

        return obs

    def get_expert_solution(self):
        return self._expert_solutions.get_solution_at(self.initial_agent_pos, self.initial_agent_dir, self._goal_pos)

    def get_expert_action(self):
        if np.random.random() < self._expert_eps:
            return self.action_space.sample()
        else:
            return self._expert_solutions.get_solution_at(self.agent_pos, self.agent_dir, self._goal_pos).actions[0]

    def next_state_hash(self, action):
        # Save neccessary state variables
        prev_agent_pos = self.agent_pos
        prev_agent_dir = self.agent_dir

        # Simulate forward and compute hash
        MiniGridEnv.step(self, action)
        fhash = self.hash()

        # Get back in previous states
        self.step_count -= 1
        self.agent_pos = prev_agent_pos
        self.agent_dir = prev_agent_dir

        return fhash

    def get_total_num_states(self):
        # agent can take 4 possible looking directions at all cells except walls
        # todo FOR FIXED width and height ONLY
        return ((8*8) * 4 - 1 + 4) * 4 * (8**4)

    def hash_grid(self, size=13):
        sample_hash = hashlib.sha256()
        to_encode = [self.grid.encode().tolist()]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]


@dataclass
class ExpertSolution:
    actions: List[MiniGridEnv.Actions]

class FourRoomsSolutions:
    def __init__(self, env: FourRoomsEnv, path=None) -> None:
        self._solutions = {}
        self._env = env

        # Load cache if specified
        if path is not None:
            with open(path, mode="rb") as f:
                self._solutions = pickle.load(f)

    def get_solution_at(self, agent_pos, agent_dir, goal_pos) -> ExpertSolution:
        t_agent_pos = tuple(agent_pos)
        t_goal_pos  = tuple(goal_pos)
        cache_key = (t_agent_pos, agent_dir, t_goal_pos, self._env.hash_grid())
        
        if cache_key not in self._solutions:
            self._solutions[cache_key] = self._compute_solution_at(agent_pos=agent_pos, agent_dir=agent_dir, goal_pos=goal_pos)

        return self._solutions[cache_key]

    def save(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self._solutions, f)
        
    def _compute_solution_at(self, agent_pos, agent_dir, goal_pos) -> ExpertSolution:
        protector = count()
        queue     = [(0, next(protector), agent_pos, agent_dir, [])]

        # Keep track of solved nodes
        solved    = set()

        # Keep track of current smallest costs
        distances = defaultdict(lambda: 100000)
        distances[tuple(agent_pos)] = 0

        while queue:
            cur_cost, _, cur_pos, cur_dir, path = heappop(queue)

            # Found a solution
            if np.array_equal(cur_pos, goal_pos):
                return ExpertSolution(path)

            # This position is already solved
            if tuple(cur_pos) in solved:
                continue

            # Update neighbors
            for neighbor_pos in self._neighbors(cur_pos):
                # To get from the cur_pos to neighbor you need to do a few actions (from 1 to 3)
                # This results in a cost + we need to keep track of actions made
                actions, result_dir = self._to_go(cur_pos, cur_dir, neighbor_pos)
                neighbor_cost       = len(actions) + cur_cost
                if neighbor_cost < distances[tuple(neighbor_pos)]:
                    distances[tuple(neighbor_pos)] = neighbor_cost
                    result = (neighbor_cost, next(protector), neighbor_pos, result_dir, path + actions)
                    heappush(queue, result)

            solved.add(tuple(cur_pos))

        raise Exception("Could not solve.")

    def _neighbors(self, pos):
        viable_neighbors = []
        for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = pos + offset

            tile_type = None
            tile      = self._env.grid.get(neighbor_pos[0], neighbor_pos[1])
            if tile is not None:
                tile_type = tile.type

            if tile_type != "wall":
                viable_neighbors.append(neighbor_pos)

        return viable_neighbors

    def _to_go(self, from_pos, from_dir, to_pos):
        if np.array_equal(from_pos + DIR_TO_VEC[from_dir], to_pos):
            return [self._env.actions.forward], from_dir
        elif np.array_equal(from_pos - DIR_TO_VEC[from_dir], to_pos):
            return [self._env.actions.left, self._env.actions.left, self._env.actions.forward], (from_dir + 2) % 4
        else:
            # Rotate right and check whether we could arrive at target position
            right_dir = (from_dir + 1) % 4
            if np.array_equal(from_pos + DIR_TO_VEC[right_dir], to_pos):
                return [self._env.actions.right, self._env.actions.forward], right_dir 
            else:
                # Otherwise rotate left
                left_dir = from_dir - 1
                if left_dir < 0:
                    left_dir += 4
                return [self._env.actions.left, self._env.actions.forward], left_dir


register(
    id='MiniGrid-FourRooms-v1',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)
