import gym
import numpy as np
from gym.utils import seeding
from gym import spaces


class PongEnv(gym.Env):

    def __init__(self):

        self.ball_max_vel = 0.1
        self.min_x_mag = 0.025
        self.paddle_max_vel = 0.1
        self.paddle_height = 1. / 8
        self.paddle_width = 1. / 60
        self.max_score = 7

        # state is tuple: (ballx, bally, ballvx, ballvy, paddley, paddlevy, score)
        self.state_low = np.array([
            0.0, 0.0, -self.ball_max_vel, -self.ball_max_vel, 0.0, -self.paddle_max_vel, 0
        ])

        self.state_high = np.array([
            1.0, 1.0, self.ball_max_vel, self.ball_max_vel, 1.0, self.paddle_max_vel, self.max_score
        ])

        self.viewer = None
        self.state = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.state_low, self.state_high)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action."

        ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, paddle_pos_y, paddle_vel_y, score = \
            self.state

        if ball_pos_x == 1.0:
            # collision with right
            ball_vel_x = -ball_vel_x

        if (ball_pos_y == 1.0 or ball_pos_y == 0.0) and ball_pos_x > 0.0:
            # collision with top or bottom
            ball_vel_y = -ball_vel_y
        
        reward = 0.0

        if ball_pos_x == 0.0:
            # collision with left (player side)

            if paddle_pos_y - self.paddle_height < ball_pos_y and paddle_pos_y + self.paddle_height > ball_pos_y:
                # collision with player paddle
                reward = 1.0                
                ball_vel_x = -ball_vel_x
            else:
                reward = -10.0
                score += 1
                ball_pos_x = 0.5
                ball_pos_y = 0.5
                ball_vel_x = self.np_random.rand() * self.ball_max_vel
                ball_vel_y = self.np_random.rand() * self.ball_max_vel
                paddle_pos_y = 0.5
                paddle_vel_y = 0.0
 
                self.state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle_pos_y,
                    paddle_vel_y,
                    score
                ]
                done = bool(score >= self.max_score)
                return np.array(self.state), reward, done, {}               


        done = bool(score >= self.max_score)
        
        ball_pos_x += ball_vel_x
        ball_pos_y += ball_vel_y
        paddle_vel_y += 0.05 * (action - 1)
        paddle_vel_y = np.clip(paddle_vel_y, -self.paddle_max_vel, self.paddle_max_vel)
        paddle_pos_y += paddle_vel_y

        ball_pos_x = np.clip(ball_pos_x, 0.0, 1.0)
        ball_pos_y = np.clip(ball_pos_y, 0.0, 1.0)
        paddle_pos_y = np.clip(paddle_pos_y, 0.0, 1.0)
        ball_vel_x = np.clip(ball_vel_x, -self.ball_max_vel, self.ball_max_vel)
        ball_vel_y = np.clip(ball_vel_y, -self.ball_max_vel, self.ball_max_vel)

        if np.abs(ball_vel_x) < self.min_x_mag:
            ball_vel_x = np.sign(ball_vel_x) * self.min_x_mag

        self.state = [
            ball_pos_x,
            ball_pos_y,
            ball_vel_x,
            ball_vel_y,
            paddle_pos_y,
            paddle_vel_y,
            score
        ]
        return np.array(self.state), reward, done, {}

    def reset(self):
        ball_pos_x = 0.5
        ball_pos_y = 0.5
        ball_vel_x = self.np_random.rand() * self.ball_max_vel
        ball_vel_y = self.np_random.rand() * self.ball_max_vel
        paddle_pos_y = 0.5
        paddle_vel_y = 0.0
        score = 0
        self.state = np.array([ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, paddle_pos_y, paddle_vel_y, score])
        return self.state
        
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = 1.0
        world_height = 1.0
        xscale = screen_width / world_width
        yscale = screen_height / world_height

        paddle_width = xscale * self.paddle_width
        paddle_height = yscale * self.paddle_height
        ball_radius = 10

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -paddle_width/2, paddle_width/2, paddle_height, 0
            paddle = rendering.FilledPolygon([
                (l,b), (l,t), (r,t), (r,b)
            ])
            paddle.set_color(0.5, 0.5, 0.5)

            self.paddle_transform = rendering.Transform()
            paddle.add_attr(self.paddle_transform)

            self.viewer.add_geom(paddle)

            ball = rendering.make_circle(ball_radius)
            ball.set_color(0.5, 0.5, 0.5)

            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)

            self.viewer.add_geom(ball)

        ball_pos_x, ball_pos_y, _, _, paddle_pos_y, _, _ = self.state
        #self.paddle_transform.set_translation(50, 50)
        self.paddle_transform.set_translation(paddle_width/2, paddle_pos_y * yscale)
        self.ball_transform.set_translation(ball_pos_x * xscale, ball_pos_y * yscale)
        #self.ball_transform.set_translation(50, 50)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = PongEnv()
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.np_random.randint(3))
        env.render()
    
    env.close()