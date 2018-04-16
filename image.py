import scipy.io
import sys,os
import numpy as np
import matplotlib
# matplotlib.use("GTKAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gfilt
from brian2 import *
from pyglet.gl import *
import pyglet

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

AGENT_IMAGE = 'agent.png'
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

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
			self.wheel_speed += np.clip(np.random.normal(0, 0.05, 2)*2, -4.0, 4.0)
			# self.wheel_speed += np.random.uniform(-1, 1, 2)
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
		self.x, self.y, self.theta, self.rotation = self.data.loc[self.time_count]
		self.time_count += int(self.dt)
		self.counter += 1
		if self.time_count == agent.duration: 
			self.time_count = 0
			self.counter = 0		



#############################################################################
# AGENT RUN
#############################################################################
duration 		= 40000 #ms
dt 				= 10 #ms
agent = Agent()
agent.run(duration, dt)

#############################################################################
# BRIAN NETWORK
#############################################################################
set_device('cpp_standalone')
start_scope()
duration = duration * ms
# defaultclock.dt = 200. * us
n = 20 # better be even number of neurons
phi = np.arange(0, 2*np.pi, 2*np.pi/n) # radial position of the neurons 
tau = 50 * ms

eqs_neurons 	= '''
dv/dt = - v/tau : 1
'''

# Firing rate noise based on agent simulation head velocity
diff = agent.data['theta'].diff(1).fillna(0)
diff[diff>np.pi] -= 2*np.pi
diff[diff<-np.pi] += 2*np.pi
angular_velocity = diff/dt
acceleration = angular_velocity.diff(1)/dt

# fake angular velocity
angular_velocity = pd.Series(index = acceleration.index.values, data = np.linspace(-1, 1, int(duration/ms/dt)))

max_firing_rate = 200
min_firing_rate = 0
# tmp = np.tanh(angular_velocity)
tmp = angular_velocity
# CW noise is diminushin when positive velocity increase (CCW turn)
CW_noise = pd.Series(index = angular_velocity.index.values, data = tmp*-1.)
CW_noise -= CW_noise.min()
CW_noise /= CW_noise.max()
CW_noise = min_firing_rate + CW_noise*(max_firing_rate - min_firing_rate)
CW_noise  = TimedArray(CW_noise.values*Hz, dt=float(dt)*ms)
# CCW noise is diminushing when negative velocity increasse (CW turn)
CCW_noise = pd.Series(index = angular_velocity.index.values, data = tmp)
CCW_noise -= CCW_noise.min()
CCW_noise /= CCW_noise.max()
CCW_noise = min_firing_rate + CCW_noise*(max_firing_rate - min_firing_rate)
CCW_noise = TimedArray(CCW_noise.values*Hz, dt=float(dt)*ms)

# Neuron group
Bckgr_CW_group	= PoissonGroup(n, rates='CW_noise(t)')
Bckgr_CCW_group	= PoissonGroup(n, rates='CCW_noise(t)')

Bckgr_CW_group	= PoissonGroup(n, rates=1000*Hz)
Bckgr_CCW_group	= PoissonGroup(n, rates=1000*Hz)

CW_group 		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
CCW_group 		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
ADN_group		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')


# background drive
Bckgr_to_CW		= Synapses(Bckgr_CW_group, CW_group, 'w : 1', on_pre='v += w')
Bckgr_to_CW.connect(i = np.arange(n), j = np.arange(n))
Bckgr_to_CW.w = 1.05
Bckgr_to_CCW	= Synapses(Bckgr_CCW_group, CCW_group, 'w : 1', on_pre='v += w')
Bckgr_to_CCW.connect(i = np.arange(n), j = np.arange(n))
Bckgr_to_CCW.w = 1.05
# reciprocal inhibitory connection
w_ccw = makeWeight(phi, np.pi+0.1)
w_cw  = makeWeight(phi, np.pi-0.1)
CW_to_CW 		= Synapses(CW_group, CW_group, 'w : 1', on_pre='v += w')
CW_to_CW.connect(p = 1)
CW_to_CW.w 		= w_cw.flatten()
CCW_to_CCW 		= Synapses(CCW_group, CCW_group, 'w : 1', on_pre='v += w')
CCW_to_CCW.connect(p = 1)
CCW_to_CCW.w 	= w_ccw.flatten()
# inter layer inhibitory connection
CW_to_CCW 		= Synapses(CW_group, CCW_group, 'w : 1', on_pre='v += w')
CW_to_CCW.connect(p = 1)
CW_to_CCW.w 	= w_cw.flatten()
CCW_to_CW 		= Synapses(CCW_group, CW_group, 'w : 1', on_pre='v += w')
CCW_to_CW.connect(p = 1)
CCW_to_CW.w 	= w_ccw.flatten()
# LMN to ADN connection
CW_to_ADN 		= Synapses(CW_group, ADN_group, 'w : 1', on_pre='v += w')
CW_to_ADN.connect(i = np.arange(n), j = np.arange(n))
CW_to_ADN.w 	= 0.5
CCW_to_ADN 		= Synapses(CCW_group, ADN_group, 'w : 1', on_pre='v += w')
CCW_to_ADN.connect(i = np.arange(n), j = np.arange(n))
CCW_to_ADN.w 	= 0.5


# Spike monitor
inp_mon 		= SpikeMonitor(Bckgr_CW_group)
cw_mon 			= SpikeMonitor(CW_group)
ccw_mon 		= SpikeMonitor(CCW_group)
adn_mon 		= SpikeMonitor(ADN_group)
# statemon 		= StateMonitor(CCW_group, 'v', record = True)

run(duration, report = 'text')
######################################################################
# FIRING RATE TUNING CURVES
######################################################################
frate = pd.DataFrame(index = np.arange(0, duration/ms, dt))
for gr,nme in zip([cw_mon, ccw_mon, adn_mon], ['lmncw_', 'lmnccw_', 'adn_']):
	spikes = gr.spike_trains()
	for n in spikes.keys():
		f, bin_edges = np.histogram(spikes[n]/ms, int(duration/ms/dt), range = (0, duration/ms))
		frate[nme+str(n)] = f/dt

# separate in 3 arrays for faster display
data = {nme:frate.filter(regex=nme).values for nme in ['lmncw_*','lmnccw_*','adn_*']}




######################################################################
# LIVE DISPLAY
######################################################################
figw, figh = (400,WINDOW_HEIGHT)
window  = pyglet.window.Window(WINDOW_WIDTH+figw, WINDOW_HEIGHT)
dpi_res = 60

# set position of agent to final position
agent.counter = len(agent.data)-1
agent.time_count = agent.data.index[-1]
agent.update(dt)

# networks liveplay
fig = Figure((figw/dpi_res, figh/dpi_res), dpi = dpi_res)
axes = {}
for rect, nme in zip([[0.05,0,0.35,0.5],[0.6,0,0.35,0.5],[0.25,0.4,0.5,0.5]],['lmncw_*','lmnccw_*','adn_*']): # left, bottom, width, height
	ax = fig.add_axes(rect, projection = 'polar')
	# ax.set_ylim(0,1)
	ax.set_autoscale_on(False)
	axes[nme] = ax
	# line, = ax.plot(phi, data[nme][-1], '-')	
	bar = ax.bar(phi, data[nme][-1], 0.3, align='center')
	npos, = ax.plot(phi, np.ones_like(phi), 'o-', color = 'black')

canvas = FigureCanvasAgg(fig)
canvas.draw()
image = pyglet.image.ImageData(figw, figh, 'RGB', data = canvas.tostring_rgb(), pitch = -3*figw)

# path of the agent
fig2 = Figure((WINDOW_WIDTH/dpi_res, WINDOW_HEIGHT/dpi_res), dpi = dpi_res)
ax2 = fig2.add_axes([0,0,1,1])
ax2.set_facecolor('black')
ax2.plot(agent.data['x'], agent.data['y'], color = 'white', linewidth = 2)
ax2.set_xlim(0, WINDOW_WIDTH)
ax2.set_ylim(0, WINDOW_HEIGHT)
ax2.set_autoscale_on(False)
canvas2 = FigureCanvasAgg(fig2)
canvas2.draw()
image2 = pyglet.image.ImageData(WINDOW_WIDTH, WINDOW_HEIGHT, 'RGB', data = canvas2.tostring_rgb(), pitch = -3*WINDOW_WIDTH)


@window.event
def on_key_press(symbol, modifiers):
    pyglet.image.get_buffer_manager().get_color_buffer().save('screenshot.png')

@window.event
def on_draw():
	window.clear()
	image2.blit(0, 0)	
	image.blit(WINDOW_WIDTH,WINDOW_HEIGHT-figh)
	agent.draw()
	



pyglet.app.run()

