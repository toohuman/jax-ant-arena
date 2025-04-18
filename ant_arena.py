"""
AntArenaEnv: A multi-agent ant environment with a circular arena using pybox2d and gymnasium.
- The ants move in a circular arena and cannot escape the boundary.
- Collision detection prevents overlap with the arena boundary and other ants.

To expand to multi-agent RL, refactor to PettingZoo or use Gym multi-agent wrappers.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2BodyDef, b2_dynamicBody, b2_staticBody)

# Optional: for quick rendering (matplotlib)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# --- Simulation scale (mm) and rendering scale (pixels/mm) ---
ARENA_DIAMETER_MM = 100.0
ARENA_RADIUS_MM = ARENA_DIAMETER_MM / 2
ANT_RADIUS_MM = 0.7  # More realistic for ant body size
SCALE = 8  # pixels per mm (will be dynamically scaled in render)


def mm_per_sec_to_box2d_velocity(mm_per_s):
    """Convert mm/sec to Box2D world units (which are mm in this env)."""
    return mm_per_s


class AntArenaEnv(gym.Env):
    """
    Multi-agent ant environment in a circular arena using pybox2d.
    All ants are active and updated each step.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, n_ants=64, arena_radius=ARENA_RADIUS_MM, ant_radius=ANT_RADIUS_MM):
        self.render_mode = render_mode
        self.n_ants = n_ants
        self.arena_radius = arena_radius
        self.ant_radius = ant_radius
        self.viewer = None
        self.ax = None
        self._setup_spaces()
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.ants = []
        self.reset()

    def _setup_spaces(self):
        # Each ant: [x, y, angle, vx, vy, ang_vel]
        obs_dim = 6 * self.n_ants
        self.action_space = spaces.Box(
            low=np.tile(np.array([0, -1], dtype=np.float32), (self.n_ants, 1)),
            high=np.tile(np.array([1, 1], dtype=np.float32), (self.n_ants, 1)),
            dtype=np.float32,
            shape=(self.n_ants, 2)
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if hasattr(self, 'world') and self.world is not None:
            for body in self.world.bodies[:]:
                self.world.DestroyBody(body)
        self.ants = []
        self._create_ants()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _create_ants(self):
        self.ants = []
        for _ in range(self.n_ants):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, self.arena_radius - self.ant_radius)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            body_def = b2BodyDef()
            body_def.type = b2_dynamicBody
            body_def.position = (x, y)
            body_def.angle = np.random.uniform(0, 2 * np.pi)
            body = self.world.CreateBody(body_def)
            shape = b2CircleShape(radius=self.ant_radius)
            body.CreateFixture(b2FixtureDef(shape=shape, density=1, friction=0.3, restitution=0.1))
            self.ants.append(body)

    def _get_obs(self):
        obs = []
        for body in self.ants:
            pos = np.array(body.position)
            vel = np.array(body.linearVelocity)
            obs.extend([*pos, body.angle, *vel, body.angularVelocity])
        return np.array(obs, dtype=np.float32)

    def _constrain_to_arena(self, body):
        pos = np.array(body.position)
        dist = np.linalg.norm(pos)
        if dist > self.arena_radius - self.ant_radius:
            pos = pos / dist * (self.arena_radius - self.ant_radius)
            body.position = pos
            body.linearVelocity = (0, 0)

    def _enforce_ant_ant_no_push(self):
        n = len(self.ants)
        for i in range(n):
            for j in range(i + 1, n):
                body_a = self.ants[i]
                body_b = self.ants[j]
                delta = np.array(body_a.position) - np.array(body_b.position)
                dist = np.linalg.norm(delta)
                min_dist = 2 * self.ant_radius
                if dist < min_dist and dist > 0:
                    move = 0.5 * (min_dist - dist) * delta / dist
                    body_a.position += move
                    body_b.position -= move
                    body_a.linearVelocity = (0, 0)
                    body_b.linearVelocity = (0, 0)

    def _project_velocity_to_heading(self, body):
        v = np.array(body.linearVelocity)
        angle = body.angle
        heading = np.array([np.cos(angle), np.sin(angle)])
        speed = np.dot(v, heading)
        body.linearVelocity = speed * heading

    def step(self, actions):
        # actions: shape (n_ants, 2)
        actions = np.asarray(actions)
        for i, body in enumerate(self.ants):
            throttle = float(np.clip(actions[i, 0], 0, 1))
            turn = float(np.clip(actions[i, 1], -1, 1))
            v_forward = 20 * throttle  # mm/sec
            body.linearVelocity = mm_per_sec_to_box2d_velocity(
                v_forward * np.array([np.cos(body.angle), np.sin(body.angle)])
            )
            body.angularVelocity = 2 * turn  # rad/sec
        # Advance world
        self.world.Step(1.0 / 30, 6, 2)
        for body in self.ants:
            self._constrain_to_arena(body)
        self._enforce_ant_ant_no_push()
        for body in self.ants:
            self._project_velocity_to_heading(body)
        obs = self._get_obs()
        reward = np.zeros(self.n_ants, dtype=np.float32)  # Placeholder: zero reward for all
        terminated = np.array([False] * self.n_ants)
        truncated = np.array([False] * self.n_ants)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if not HAS_MPL:
            print("matplotlib not installed: cannot render")
            return
        fig_size = 10
        pixel_window = 1000
        arena_pixel_radius = int(pixel_window * 0.5)
        scale = arena_pixel_radius / self.arena_radius
        if self.viewer is None:
            self.viewer = plt.figure(figsize=(fig_size, fig_size))
            self.ax = self.viewer.add_subplot(1, 1, 1)
        self.ax.clear()
        circle = plt.Circle((0, 0), self.arena_radius * scale, color='lavender', fill=True, zorder=0)
        self.ax.add_patch(circle)
        r = self.ant_radius * scale
        for body in self.ants:
            pos = body.position
            angle = body.angle
            triangle = np.array([
                [r, 0],
                [-r * 0.6, r * 0.5],
                [-r * 0.6, -r * 0.5],
            ])
            rot = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            triangle_rot = (triangle @ rot.T) + np.array([pos.x * scale, pos.y * scale])
            ant_tri = plt.Polygon(triangle_rot, color='blue', zorder=2)
            self.ax.add_patch(ant_tri)
        lim = self.arena_radius * scale
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_position([0, 0, 1, 1])
        self.viewer.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.pause(0.001)

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None


def main():
    """
    Run a simple test loop for the AntArenaEnv with matplotlib visualization.
    All ants move forward with a small random turn.
    """
    env = AntArenaEnv(render_mode='human', n_ants=64)
    obs, info = env.reset()
    for step in range(300):
        # Each ant: random throttle [0,0.5], random turn [-0.3,0.3]
        actions = np.stack([
            np.random.uniform(0, 0.5, size=env.n_ants),
            np.random.uniform(-0.3, 0.3, size=env.n_ants)
        ], axis=1)
        obs, reward, terminated, truncated, info = env.step(actions)
        env.render()
        if np.any(terminated) or np.any(truncated):
            print(f"Episode finished after {step+1} steps.")
            break
    env.close()


if __name__ == "__main__":
    main()