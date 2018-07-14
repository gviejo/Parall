import scipy.io
import sys,os
import numpy as np
import pandas as pd

import pyglet

class Agent(pyglet.sprite.Sprite):
	AGENT_IMAGE 			= 'agent.png'
	agent_image             = pyglet.resource.image(AGENT_IMAGE)
	agent_image.anchor_x    = agent_image.width / 2
	agent_image.anchor_y    = agent_image.height / 2
	width                   = agent_image.width
	height                  = agent_image.height

	def __init__(self, WINDOW_WIDTH, WINDOW_HEIGHT):
		self.WINDOW_WIDTH	= WINDOW_WIDTH
		self.WINDOW_HEIGHT 	= WINDOW_HEIGHT
		x                   = WINDOW_WIDTH/2.
		y                   = WINDOW_HEIGHT/2.
		r                   = np.pi + np.pi/2 # radians
		super(Agent, self).__init__(self.agent_image, x, y)
		self.theta          = np.pi/2.
		self.rotation       = -1*self.theta*180/np.pi
		self.wheel_speed    = np.ones(2)*25.0
		self.time_count = 0
		self.counter = 0

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
			self.wheel_speed += np.clip(np.random.normal(0, 0.005, 2), -0.5, 0.5)
			vl, vr 		= self.wheel_speed
			cste1 		= (vr + vl)/2.
			theta 		+= dt*((vr - vl)/2.)
			if theta < 0.0 : theta += 2*np.pi
			theta 		%= 2*np.pi
			x 		 	+= dt*(cste1*np.cos(theta))
			y           += dt*(cste1*np.sin(theta))
			rotation    = -1*theta*180/np.pi
			x           = min(max(x, self.width), self.WINDOW_WIDTH - self.width/2)
			y           = min(max(y, self.height), self.WINDOW_HEIGHT - self.height/2)        	
			self.data.loc[t] = np.array([x, y, theta, rotation])
		
	def update(self, dt):
		self.x, self.y, self.theta, self.rotation = self.data.loc[self.time_count]
		self.time_count += int(self.dt)
		self.counter += 1
		if self.time_count == self.duration: 
			self.time_count = 0
			self.counter = 0


def makeInhibitionWeight(n):
	w = np.ones((n*n,n*n))
	for i in range(n):
		w[i*n:i*n+n,i*n:i*n+n] = np.zeros((n,n))
	# w = np.tile(np.eye(n,n), (n,n))
	# for i in range(n):
	# 	w[i*n:i*n+n,i*n:i*n+n] = np.zeros((n,n))
	return w

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)    
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

def makeWeight(phi, offset):
	ang_dist = (np.vstack(phi) - phi)*-1.0
	ang_dist[ang_dist<0.0] += 2*np.pi
	w = ang_dist.copy()
	if offset < np.pi: # clockwise rotation
		index = np.logical_and(ang_dist>offset, ang_dist<(2*np.pi - 2*(np.pi-offset)))
		w[index] = offset - np.abs(offset - ang_dist[index]) # lower quadrant
		w[ang_dist>(2*np.pi-2*(np.pi-offset))] = 0.0
	elif offset > np.pi: # counter-clockwise rotation
		w[ang_dist<2*(offset-np.pi)] = 0.0		
		index = np.logical_and(ang_dist < offset, ang_dist > 2*(offset-np.pi))
		w[index] = ang_dist[index] - 2*(offset-np.pi)
		w[ang_dist>offset] = np.abs(ang_dist[ang_dist>offset] - 2*np.pi)
	w = np.cos(w)-1
	w /= np.abs(w.min())
	return w

def makePlaceFields(center_pc, cov_, width, height):
	n_pc = len(center_pc)
	to_return = np.zeros((n_pc, width, height))
	xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
	for i, (x,y) in enumerate(center_pc):
		tmp = np.exp(-np.power(xx - x, 2.0)*cov_) * np.exp(-np.power(yy - y, 2.0)*cov_)
		tmp[tmp < 0.2] = 0.0
		to_return[i] = tmp
	return to_return

def computeDistanceActivation(pos_xy, center_pc, cov_):
	# euclidean distance
	# d = np.sqrt(np.power(np.vstack(pos_xy[:,0]) - center_pc[:,0], 2.0) + np.power(np.vstack(pos_xy[:,1]) - center_pc[:,1], 2.0))
	# d = 1./(1. + np.exp(-d))
	# d[d<0.8] = 0.0
	# return d
	# gaussian distance
	x = np.exp(-np.power(np.vstack(pos_xy[:,0]) - center_pc[:,0], 2.0)*cov_)
	y = np.exp(-np.power(np.vstack(pos_xy[:,1]) - center_pc[:,1], 2.0)*cov_)
	d = x * y
	d[d < 0.2] = 0.0
	d[d > 0.2] = 1.0
	# d = 1./(1. + np.exp(-(d - 0.15)*10.))
	return d

def computeCueActivation(theta, theta_cue, vphi):
	tmp = theta - theta_cue
	tmp += 2*np.pi
	tmp %= 2*np.pi
	tmp2 = np.vstack(tmp) - vphi
	tmp3 = np.cos(tmp2)
	return 1./(1. + np.exp(-(tmp3 - 0.75)*10.))	