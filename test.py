import scipy.io
import sys,os
import numpy as np
from matplotlib.pyplot import plot, ion, show
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gfilt
from pyglet.gl import *
import pyglet

ion()

AGENT_IMAGE = 'agent.png'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class Agent(pyglet.sprite.Sprite):
	agent_image             = pyglet.resource.image(AGENT_IMAGE)
	agent_image.anchor_x    = agent_image.width / 2
	agent_image.anchor_y    = agent_image.height / 2
	width                   = agent_image.width
	height                  = agent_image.height

	def __init__(self):
		x                   = WINDOW_WIDTH/2.
		y                   = WINDOW_HEIGHT/2.
		r                   = np.pi + np.pi/2 # radians
		super(Agent, self).__init__(self.agent_image, x, y)
		self.theta          = np.pi/2.
		self.rotation       = -1*self.theta*180/np.pi
		self.wheel_speed    = np.ones(2)*20.0
		self.time_count = 0

	def run(self, duration, dt):		
		self.duration = duration
		self.dt 		= dt
		self.times 	= np.arange(0, duration, dt)
		self.data 	= pd.DataFrame(index = self.times, columns = ['x', 'y', 'theta', 'rotation'], data = 0)			
		theta 		= self.theta
		x 			= self.x
		y 			= self.y
		dt 			= dt/1000. 		
		for i, t in enumerate(self.times):
			self.wheel_speed += np.random.normal(0, 0.1, 2)
			vl, vr 		= self.wheel_speed
			cste1 		= (vr + vl)/2.
			theta 		+= dt*((vr - vl)/2.)
			if theta < 0.0 : theta += 2*np.pi
			theta 		%= 2*np.pi
			x 		 	+= dt*(cste1*np.cos(theta))
			y           += dt*(cste1*np.sin(theta))
			rotation    = -1*theta*180/np.pi
			x           = min(max(x, self.width), WINDOW_WIDTH - self.width/2)
			y           = min(max(y, self.height), WINDOW_HEIGHT - self.height/2)        	
			self.data.loc[t] = np.array([x, y, theta, rotation])
		
	def update(self, dt):
		if self.time_count == agent.duration: self.time_count = 0
		self.x, self.y, self.theta, self.rotation = self.data.loc[self.time_count]
		self.time_count += int(self.dt)

#############################################################################
# AGENT RUN
#############################################################################
duration 		= 10000 #ms
dt 				= 10 #ms
agent = Agent()
agent.run(duration, dt)


######################################################################
# LIVE DISPLAY
######################################################################
window  = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

def update(dt):
    agent.update(dt)

@window.event
def on_draw():
    window.clear()
    agent.draw()



pyglet.clock.schedule_interval(update, 1/1000.)
pyglet.app.run()

window.close()