import os
import sys
import numpy as np

from pyglet.gl import *
import pyglet

AGENT_IMAGE = 'agent.png'


class Agent(pyglet.sprite.Sprite):
    agent_image             = pyglet.resource.image(AGENT_IMAGE)
    agent_image.anchor_x    = agent_image.width / 2
    agent_image.anchor_y    = agent_image.height / 2
    width                   = agent_image.width
    height                  = agent_image.height

    def __init__(self):
        x                   = window.width/2.
        y                   = window.height/2.
        r                   = np.pi + np.pi/2 # radians
        super(Agent, self).__init__(self.agent_image, x, y)
        self.theta          = np.pi/2.
        self.rotation       = -1*self.theta*180/np.pi
        self.wheel_speed    = np.ones(2)*20.0

    def update(self, dt):
        #print(self.x, self.y, self.theta, self.rotation)
        print(self.wheel_speed)
        self.wheel_speed    += np.random.normal(0,0.01,2)
        vl, vr              = self.wheel_speed
        cste1               = (vr + vl)/2.
        self.theta          += dt*((vr-vl)/2.)
        if self.theta < 0.0 : self.theta += 2*np.pi
        self.theta          %= 2*np.pi
        self.x              += dt*(cste1*np.cos(self.theta))
        self.y              += dt*(cste1*np.sin(self.theta))
        self.rotation       = -1*self.theta*180/np.pi
        self.x              = min(max(self.x, self.width), window.width - self.width/2)
        self.y              = min(max(self.y, self.height), window.height - self.height/2)


window  = pyglet.window.Window(800, 600)
agent   = Agent()
positions = []


def update(dt):
    agent.update(dt)
    positions.append([agent.x, agent.y, agent.theta])

@window.event
def on_draw():
    window.clear()
    agent.draw()

pyglet.clock.schedule_interval(update, 1/100.)
pyglet.app.run()
