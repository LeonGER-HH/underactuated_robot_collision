import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pygame as pg
import helpers.comp_geometry as cgeo


class Obstacle:
    """circular obstacle"""
    obstacles = []

    def __init__(self, x, y, r):
        self.center = np.array([x, y])
        self.radius = r
        Obstacle.obstacles.append(self)

        self.dist_log = []


class Robot:
    robots = []

    def __init__(self, x0, u, color):
        self.x0 = x0  # x, y, theta, dtheta
        self.x = self.x0.copy()
        self.I = 1 / 36 * 2 * 3**3
        self.m = 1
        self.u = u

        Robot.robots.append(self)

        self.x_bag = x0.reshape((4, 1))
        # for plotting purposes
        self.trail = self.x[:2].copy().reshape((1, 2))
        self.color = color

    def f_simulate(self, u, dt):
        # collision check
        dist = np.inf
        p_contact = None
        obs = None
        for obs_ in Obstacle.obstacles:
            dist_, p_contact_ = cgeo.polygon_circle_distance(self.get_vertices(), obs_.center, obs_.radius)
            if dist > dist_:
                dist, p_contact, obs = dist_, p_contact_, obs_
        # if collison, calculate impact force etc.
        eps = 1E-2
        if dist <= eps:
            theta = self.x[2]
            v_r1 = self.u * np.array([np.sin(theta), np.cos(theta), 0.])
            r_rp = p_contact - self.x[:2]
            omega_r1 = np.array([0., 0., self.x[3]])
            v_rp1 = v_r1 + np.cross(omega_r1, r_rp)
            n = p_contact - obs.center
            n = n / np.linalg.norm(n)
            n = np.append(n, 0)
            e = 1.
            j = -(1 + e) * np.dot(v_rp1, n) / (1/self.m + np.cross(r_rp, n)**2 / self.I)
            v_r2 = v_r1 + j * n / self.m
            omega_r2 = omega_r1 + np.cross(r_rp, j*n) / self.I

            # update states
            self.x[0] += v_r2[0] * dt
            self.x[1] += v_r2[0] * dt
            self.x[2] += omega_r2[2] * dt
            self.x[3] = omega_r2[2]


        else:
            # update states
            self.x[0] += np.cos(self.x[2]) * u * dt
            self.x[1] += np.sin(self.x[2]) * u * dt
            self.x[2] += self.x[3] * dt
            mu = 0.5
            if self.x[3] > 0:
                self.x[3] = max(0., self.x[3] - mu)
            else:
                self.x[3] = min(0., self.x[3] + mu)

        self.x_bag = np.hstack([self.x_bag, self.x.reshape((4, 1))])

    def get_vertices(self):
        s = 1
        #v_rel = s * np.array([[-1, 3], [1, 3], [1, 0], [3, 0], [3, -1], [-1, -1]])
        v_rel = s * np.array([[-1, 1], [2, 0], [-1, -1]])
        x = self.x.copy()
        rot_mat = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])
        v = np.ones((v_rel.shape[0], 1)) @ x[:2].reshape((1, 2)) + (rot_mat @ v_rel.T).T
        return v


x0 = np.array([2., 2., np.pi/2, 0.])
robot1 = Robot(x0, 5, (166, 206, 227))
robot2 = Robot(x0, 10, (178, 223, 138))
Obstacle(3, 3, .3)
Obstacle(6, 5.7, .3)

"""visualization"""
screen_width = 900
screen_height = 900
sf = 30


def cc2pgc(cc):
    #transform the computational coordinates to pycharm coordinates
    pgc = cc @ np.array([[sf, 0], [0, -sf]]) + np.array([[screen_width * 1/4, screen_height * 3/4]])
    return pgc.astype(int)


pg.init()
screen = pg.display.set_mode((screen_width, screen_height))
pg.display.set_caption("Contact inclusive forward simulation of an underactuated triangle robot")
running = True
time = 0
dt = 0.01
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # Fill the screen with a white background
    screen.fill("white")

    # draw grid lines
    interval = 1
    env_size = 25
    for xi in np.arange(0, env_size, interval):
        pg.draw.line(screen, "grey", cc2pgc(np.array([xi, 0]))[0], cc2pgc(np.array([xi, env_size]))[0], 1)
    for yi in np.arange(0, env_size, interval):
        pg.draw.line(screen, "grey", cc2pgc(np.array([0, yi]))[0], cc2pgc(np.array([env_size, yi]))[0], 1)


    # Draw the coordinate system
    pg.draw.line(screen, "red", cc2pgc(np.array([0, 0]))[0], cc2pgc(np.array([1, 0]))[0], 2)
    pg.draw.line(screen, "red", cc2pgc(np.array([0, 0]))[0], cc2pgc(np.array([0, 1]))[0], 2)

    #Draw Legend
    font = pg.font.Font(None, 30)
    legend_start = cc2pgc(np.array([2, 9]))[0]
    offset = 30  # Offset between items
    for i in range(len(Robot.robots)):
        robot = Robot.robots[i]
        pg.draw.rect(screen, robot.color, (legend_start[0] + i * offset, legend_start[1], 30, 30))
        label = font.render("u = " + (1-i)*"  " + str(robot.u) + "m/s", True, (0, 0, 0))
        label = pg.transform.rotate(label, 90)
        screen.blit(label, (legend_start[0] + 5 + i * offset, legend_start[1] - 100))
    pg.draw.circle(screen, (0, 0, 0), cc2pgc(np.array([1.5, 8.5]))[0], sf * 0.3)
    label = font.render("Obstacle", True, (0, 0, 0))
    label = pg.transform.rotate(label, 90)
    screen.blit(label, (cc2pgc(np.array([1, 9]))[0][0] + 5, cc2pgc(np.array([1, 9]))[0][1] - 93))


    # Draw obstacles
    for obs in Obstacle.obstacles:
        pg.draw.circle(screen, (0, 0, 0), cc2pgc(obs.center)[0], sf * obs.radius)

    for robot in Robot.robots:
        # Draw the robot
        v_robot = cc2pgc(robot.get_vertices())
        robot.trail = np.vstack([robot.trail, robot.x[:2].reshape((1, 2))])
        pg.draw.polygon(screen, robot.color, v_robot)
        # Update the robot
        u = 10
        robot.f_simulate(u, dt)

        for p in robot.trail:
            pg.draw.circle(screen, robot.color, cc2pgc(p)[0], 2)

        if robot.x[0] >= 20 or robot.x[1] >= 20:
            robot.x = robot.x0.copy()

    pg.time.wait(int(dt * 1000))

    # Update the display
    pg.display.flip()
    time += dt

# Quit Pygame
pg.quit()
plt.plot(robot2.x_bag[2, :], label="theta")
plt.plot(robot2.x_bag[3, :], "o", label="theta dot")
plt.legend()
plt.show()

