
import hashlib
import pickle
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Any, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
from gym_minigrid.envs.fourroomsv1 import (COLOR_TO_IDX, DIR_TO_VEC,
                                                OBJECT_TO_IDX, FourRoomsEnv,
                                                MiniGridEnv)
from gym_minigrid.minigrid import Grid

from policies.policy import Policy


class FourRoomsExpertPolicy(Policy):
    def __init__(
        self,
        expert_eps: float, 
        env: FourRoomsEnv, 
        solutions_path: Optional[str]
    ):
        self._expert_eps = expert_eps
        self._env        = env
        self._solutions  = FourRoomsSolutions(
            env=env,
            path=solutions_path
        )

    def set_expert_eps(self, expert_eps: float) -> None:
        self._expert_eps = expert_eps

    def predict_actions(self, obs: List[np.ndarray]) -> List[MiniGridEnv.Actions]:
        actions = []
        for iobs in obs:
            if np.random.random() < self._expert_eps:
                actions.append(self._env.action_space.sample())
            else:
                agent_pos, agent_dir, goal_pos, grid = self._decode_observation(iobs)
                solution = self._solutions.get_solution_at(agent_pos, agent_dir, goal_pos, grid)
                actions.append(solution.actions[0])

        return actions

    def _decode_observation(self, obs: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, Grid]:
        channels, width, height = obs.shape
        assert channels == 3

        agent_dir: Optional[int]        = None
        agent_pos: Optional[np.ndarray] = None
        goal_pos:  Optional[np.ndarray] = None

        for w in range(width):
            for h in range(height):
                tile = obs[:, w, h]

                # Check for the agent
                if tile[0] == OBJECT_TO_IDX["agent"] and tile[1] == COLOR_TO_IDX["red"]:
                    agent_pos = np.array([w, h])
                    agent_dir = int(tile[-1])

                # Check for the goal position
                if tile[0] == OBJECT_TO_IDX["goal"] and tile[1] == COLOR_TO_IDX["green"]:
                    goal_pos = np.array([w, h])

        if agent_dir is None:
            raise Exception("Could not find the agent.")
        if agent_pos is None:
            raise Exception("Could not find the agent.")
        if goal_pos is None:
            raise Exception("Could not find the goal.")

        # Remove agent from the observation (needed for MiniGrid.decode)
        cobs                                = deepcopy(obs)
        cobs[:, agent_pos[0], agent_pos[1]] = np.array([OBJECT_TO_IDX["empty"], 0, 0])
        cobs.astype(dtype="uint8", copy=False)
        grid, _ = Grid.decode(cobs.transpose(1, 2, 0))

        return agent_pos, agent_dir, goal_pos, grid


@dataclass
class ExpertSolution:
    actions: List[MiniGridEnv.Actions]


class FourRoomsSolutions:
    def __init__(self, env: FourRoomsEnv, path: Optional[str] = None) -> None:
        self._solutions   = {}
        self._env         = env

        # Load cache if specified
        if path is not None:
            with open(path, mode="rb") as f:
                self._solutions = pickle.load(f)

    def get_solution_at(
        self, 
        agent_pos: np.ndarray,
        agent_dir: int, 
        goal_pos:  np.ndarray,
        grid:      Grid
    ) -> ExpertSolution:
        t_agent_pos = tuple(agent_pos)
        t_goal_pos  = tuple(goal_pos)
        grid_hash   = hash_grid(grid)
        cache_key   = (t_agent_pos, agent_dir, t_goal_pos, grid_hash)
        
        if cache_key not in self._solutions:
            self._solutions[cache_key] = self._compute_solution_at(
                agent_pos=agent_pos, agent_dir=agent_dir, goal_pos=goal_pos, grid=grid
            )

        return self._solutions[cache_key]

    def save(self, path: str):
        with open(path, mode="wb") as f:
            pickle.dump(self._solutions, f)
        
    def _compute_solution_at(
        self, 
        agent_pos: np.ndarray,
        agent_dir: int, 
        goal_pos:  np.ndarray,
        grid:      Grid
    ) -> ExpertSolution:
        protector        = count()
        queue: List[Any] = [(0, next(protector), agent_pos, agent_dir, [])]

        # Keep track of solved nodes
        solved = set()

        # Keep track of current smallest costs
        distances                     = defaultdict(lambda: 100000)
        distances[tuple(agent_pos)]   = 0

        while queue:
            cur_cost, _, cur_pos, cur_dir, path = heappop(queue)

            # Found a solution
            if np.array_equal(cur_pos, goal_pos):
                return ExpertSolution(path)

            # This position is already solved
            if tuple(cur_pos) in solved:
                continue

            # Update neighbors
            for neighbor_pos in self._neighbors(cur_pos, grid):
                # To get from the cur_pos to neighbor you need to do a few actions (from 1 to 3)
                # This results in a cost + we need to keep track of actions made
                actions, result_dir = self._to_go(cur_pos, cur_dir, neighbor_pos)
                neighbor_cost       = len(actions) + cur_cost
                if neighbor_cost < distances[tuple(neighbor_pos)]:
                    distances[tuple(neighbor_pos)] = neighbor_cost
                    result              = (neighbor_cost, next(protector), neighbor_pos, result_dir, path + actions)
                    heappush(queue, result)

            solved.add(tuple(cur_pos))

        raise Exception("Could not solve.")

    def _neighbors(
        self, 
        pos: np.ndarray, 
        grid: Grid
    ) -> List[np.ndarray]:
        viable_neighbors = []
        for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = pos + offset

            tile_type = None
            tile      = grid.get(neighbor_pos[0], neighbor_pos[1])
            if tile is not None:
                tile_type = tile.type

            if tile_type != "wall":
                viable_neighbors.append(neighbor_pos)

        return viable_neighbors

    def _to_go(
        self, 
        from_pos: np.ndarray, 
        from_dir: int, 
        to_pos: np.ndarray
    ) -> Tuple[List[MiniGridEnv.Actions], int]:
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


def hash_grid(grid: Grid, size: int = 13) -> str:
    sample_hash   = hashlib.sha256()
    to_encode     = [grid.encode().tolist()]
    for item in to_encode:
        sample_hash.update(str(item).encode('utf8'))

    return sample_hash.hexdigest()[:size]
