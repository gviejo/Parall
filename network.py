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
from functions import *

import pyglet

#############################################################################
# AGENT RUN
#############################################################################


WINDOW_WIDTH 	= 500
WINDOW_HEIGHT 	= 500	
duration 		= 40000 #ms
dt 				= 10 #ms
agent = Agent(WINDOW_WIDTH, WINDOW_HEIGHT)
agent.run(duration, dt)


#############################################################################
# BRIAN NETWORK
#############################################################################
set_device('cpp_standalone')
start_scope()
duration = duration * ms

n = 20 # better be even number of neurons
n_pc = 16 # number of place cells
phi = np.arange(0, 2*np.pi, 2*np.pi/n) # radial position of the neurons 
# center_pc = np.hstack((np.random.uniform(0, WINDOW_WIDTH, (n_pc,1)), np.random.uniform(0, WINDOW_HEIGHT, (n_pc,1))))
center_pc = np.array(np.meshgrid(np.linspace(0, WINDOW_WIDTH, int(np.sqrt(n_pc))), np.linspace(0, WINDOW_HEIGHT, int(np.sqrt(n_pc))))).reshape(2, n_pc).T
tau = 25 * ms


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
# angular_velocity = pd.Series(index = acceleration.index.values, data = np.linspace(-1, 1, int(duration/ms/dt)))
# <angular_velocity = pd.Series(index = acceleration.index.values, data = np.ones(len(angular_velocity))*0.5)

# place cells activation
sigma = 0.0002
PC_activation = computeDistanceActivation(agent.data[['x','y']].values, center_pc, sigma)
PC_activation = PC_activation*10.0 # Hz
PC_activation = TimedArray(PC_activation*Hz, dt=float(dt)*ms)

# VISUAL activation
cue_pos = np.array([WINDOW_WIDTH/2,WINDOW_HEIGHT])
vphi = np.arange(0, 2*np.pi, 2*np.pi/n)
VIS_activation = computeCueActivation(agent.data['theta'], np.pi/2, vphi)
VIS_activation = VIS_activation * 10.0 # Hz
VIS_activation = TimedArray(VIS_activation*Hz, dt=float(dt)*ms)

max_firing_rate = 10000
min_firing_rate = 0
# tmp = np.tanh(angular_velocity)
tmp = angular_velocity
# CW noise is diminushing when positive velocity increase (CCW turn)
CW_noise = pd.Series(index = angular_velocity.index.values, data = tmp*-1.)
CW_noise -= CW_noise.min()
CW_noise /= CW_noise.max()
CW_noise = CW_noise.fillna(0)
CW_noise = min_firing_rate + CW_noise*(max_firing_rate - min_firing_rate)
# CW_noise = pd.Series(index = angular_velocity.index.values, data = np.ones(len(angular_velocity))*(max_firing_rate/2.))
CW_noise  = TimedArray(CW_noise.values*Hz, dt=float(dt)*ms)
# CCW noise is diminushing when negative velocity increase (CW turn)
CCW_noise = pd.Series(index = angular_velocity.index.values, data = tmp)
CCW_noise -= CCW_noise.min()
CCW_noise /= CCW_noise.max()
CCW_noise = CCW_noise.fillna(0)
CCW_noise = min_firing_rate + CCW_noise*(max_firing_rate - min_firing_rate)
# CCW_noise = pd.Series(index = angular_velocity.index.values, data = np.ones(len(angular_velocity))*(max_firing_rate/2.))
CCW_noise = TimedArray(CCW_noise.values*Hz, dt=float(dt)*ms)

###########################################################################################################
# Neuron group
###########################################################################################################
Bckgr_CW_group	= PoissonGroup(n, rates='CW_noise(t)')
Bckgr_CCW_group	= PoissonGroup(n, rates='CCW_noise(t)')
PC_group 		= PoissonGroup(n_pc, rates = 'PC_activation(t,i)')
VIS_group 		= PoissonGroup(n, rates = 'VIS_activation(t, i)')
CW_group 		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
CCW_group 		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
ADN_group		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
POS_group		= NeuronGroup(n, model=eqs_neurons, threshold='v>1', reset='v=0', method = 'exact')
RSC_group 		= NeuronGroup(n*n_pc, model=eqs_neurons, threshold='v>1', reset = 'v=0', method = 'exact')


###########################################################################################################
# SYNAPSES
###########################################################################################################
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
#ADN to POS connection
ADN_to_POS 		= Synapses(ADN_group, POS_group, 'w :1', on_pre='v += w')
ADN_to_POS.connect(i = np.arange(n), j = np.arange(n))
ADN_to_POS.w 	= 1.0
#PC to RSC connection
PC_to_RSC 		= Synapses(PC_group, RSC_group, 'w :1', on_pre='v += w')
PC_to_RSC.connect(i = np.repeat(np.arange(n_pc), n), j = np.arange(n*n_pc))
PC_to_RSC.w 	= 0.4
#RSC to RSC connection
RSC_to_RSC 		= Synapses(RSC_group, RSC_group, 'w :1', on_pre='v += w')
RSC_to_RSC.connect(p=1)
RSC_to_RSC.w 	= makeInhibitionWeight(n)*-40.0
#ADN to RSC connection 
ADN_to_RSC 		= Synapses(ADN_group, RSC_group, 'w :1', on_pre='v += w')
ADN_to_RSC.connect(i = np.tile(np.arange(n), n_pc), j = np.arange(n*n_pc))
ADN_to_RSC.w 	= 0.2
#RSC to POS connection
RSC_to_POS 		= Synapses(RSC_group, POS_group, 'w :1', on_pre='v += w')
RSC_to_POS.connect(i = np.arange(n*n_pc), j = np.tile(np.arange(n), n_pc))
RSC_to_POS.w 	= 0.5
#POS to CW connection
POS_to_CW 		= Synapses(POS_group, CW_group, 'w :1', on_pre='v += w')
POS_to_CW.connect(i = np.arange(n), j = np.arange(n))
POS_to_CW.w 	= 1.0
#POS to CCW connection
POS_to_CCW 		= Synapses(POS_group, CCW_group, 'w :1', on_pre='v += w')
POS_to_CCW.connect(i = np.arange(n), j = np.arange(n))
POS_to_CCW.w 	= 1.0
#VIS to RSC connection
taupre = taupost = 20*ms
wmax = 0.01
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05
VIS_to_RSC 		= Synapses(VIS_group, RSC_group, 
	'''
	w : 1
	dapre/dt = -apre/taupre : 1 (event-driven)
	dapost/dt = -apost/taupost : 1 (event-driven)
	''',
	on_pre='''
	v += w
    apre += Apre
    w = clip(w+apost, 0, wmax)
	''',
	on_post='''
	apost += Apost
	w = clip(w+apre, 0, wmax)
	''')
VIS_to_RSC.connect(p = 1)
VIS_to_RSC.w 	= 0.0


###########################################################################################################
# Spike monitor
###########################################################################################################
inp_mon 		= SpikeMonitor(Bckgr_CW_group)
cw_mon 			= SpikeMonitor(CW_group)
ccw_mon 		= SpikeMonitor(CCW_group)
adn_mon 		= SpikeMonitor(ADN_group)
pos_mon			= SpikeMonitor(POS_group)
pc_mon 			= SpikeMonitor(PC_group)
rsc_mon			= SpikeMonitor(RSC_group)
vis_mon 		= SpikeMonitor(VIS_group)

###########################################################################################################
# RUN
###########################################################################################################
run(duration, report = 'text')

######################################################################
# FIRING RATE TUNING CURVES
######################################################################
frate = pd.DataFrame(index = np.arange(0, duration/ms, dt))
for gr,nme in zip([cw_mon, ccw_mon, adn_mon, pos_mon], ['lmncw_', 'lmnccw_', 'adn_', 'pos_']):
	spikes = gr.spike_trains()
	for neur in spikes.keys():
		f, bin_edges = np.histogram(spikes[neur]/ms, int(duration/ms/dt), range = (0, duration/ms))
		frate[nme+str(neur)] = f/dt

# separate in arrays for faster display
data = {nme:frate.filter(regex=nme).values for nme in ['lmncw_*','lmnccw_*','adn_*', 'pos_*']}



######################################################################
# LIVE DISPLAY
######################################################################
figw, figh = (400,WINDOW_HEIGHT)
window  = pyglet.window.Window(WINDOW_WIDTH+figw, WINDOW_HEIGHT)
dpi_res = 60

# networks liveplay
fig = Figure((figw/dpi_res, figh/dpi_res), dpi = dpi_res)
axes = {}
for rect, nme in zip([[0.05,0,0.35,0.5],[0.6,0,0.35,0.5],[0.25,0.5,0.5,0.5]],['lmncw_*','lmnccw_*','adn_*']): # left, bottom, width, height
	ax = fig.add_axes(rect, projection = 'polar')
	ax.set_ylim(0,0.5)
	ax.set_autoscale_on(False)
	axes[nme] = ax
canvas = FigureCanvasAgg(fig)
canvas.draw()
backgrounds = {}
lines = {}
neurons_position = {}
bars = {}
for nme, ax in axes.items():	
	line, = ax.plot(phi, data[nme][0], '-')	
	npos, = ax.plot(phi, np.ones_like(phi)*0.4, 'o', color = 'black')
	backgrounds[nme] = canvas.copy_from_bbox(ax.bbox)
	lines[nme] = line
	neurons_position[nme] = npos

image = pyglet.image.ImageData(figw, figh, 'RGB', data = canvas.tostring_rgb(), pitch = -3*figw)


def update(dt):	
	print(dt)
	window.clear()
	agent.update(dt)
	for nme in ['lmncw_*','lmnccw_*','adn_*']:
		canvas.restore_region(backgrounds[nme])
		lines[nme].set_ydata(data[nme][agent.counter])
		axes[nme].draw_artist(lines[nme])				
		axes[nme].draw_artist(neurons_position[nme])
		canvas.blit(axes[nme].bbox)


	image.set_data(data = canvas.tostring_rgb(), format = 'RGB', pitch = -3*figw)
	image.blit(WINDOW_WIDTH,WINDOW_HEIGHT-figh)
	# path_image.blit(0,0)	
	agent.draw()

@window.event
def on_draw():			
	# agent.draw()    
	pass

# pyglet.clock.schedule_interval(update, 1/100.)
# pyglet.app.run()
window.close()


#######################################################################
# PLOT
#######################################################################
ion()
theta = agent.data['theta']
bins = np.hstack([phi, np.array([2*np.pi])])
pos_theta = pd.Series(index = theta.index.values, data = np.digitize(theta, bins)-1)

figure()
plot(pos_mon.t/ms, pos_mon.i+3*n+3, '.y', label = 'POs')
plot(adn_mon.t/ms, adn_mon.i+2*n+2, '.g', label = 'ADN')
plot(cw_mon.t/ms, cw_mon.i+n+1, '.r', label = 'LMN(clockwise)')
plot(ccw_mon.t/ms, ccw_mon.i, '.b', label = 'LMN(counter clockwise)')
legend()

figure()
subplot(211)
plot(PC_activation.values)
subplot(212)
idx_rsc =np.array_split(RSC_group.indices[:], n_pc)
for i in range(n_pc): 	
	plot(pc_mon.t[pc_mon.i == i]/ms, pc_mon.i[pc_mon.i == i]+(i*len(idx_rsc[i])+1), 'o', label = str(i))	
	for j in idx_rsc[i]:
		plot(rsc_mon.t[rsc_mon.i == j]/ms, rsc_mon.i[rsc_mon.i == j] + i + 2, '.', color = 'black')	




tofrate = pd.DataFrame(index = frate.index, columns = phi, data = 0)
for nme in ['lmncw_*','lmnccw_*','adn_*']: tofrate += frate.filter(regex=nme).values
# tofrate = tofrate.rolling(window=10, win_type ='gaussian', center =True, axis = 0).mean(std = 1).fillna(0)
tofrate = tofrate.rolling(window=20, win_type ='triang', center =True, axis = 0).mean().fillna(0)
estim_phi = pd.Series(index = tofrate.index, data = phi[np.argmax(tofrate.values, axis = 1)])
diff = estim_phi.diff(1).fillna(0)
diff[diff>np.pi] -= 2*np.pi
diff[diff<-np.pi] += 2*np.pi
estim_vel = diff/dt
figure()
subplot(211)
plot(CW_noise.values, estim_vel.values, '-', label = 'cw')
legend()
subplot(212)
plot(CCW_noise.values, estim_vel.values, '-', label= 'ccw')
legend()

figure()
plot(vis_mon.t/ms, vis_mon.i, '.b')
title('V')

figure()
stdp = np.array(VIS_to_RSC.w).reshape(n, n_pc, n) # (VIS, PC, RSC)
for i in arange(n_pc):
	ax = subplot(int(np.sqrt(n_pc)), int(np.sqrt(n_pc)), i+1)#, projection = 'polar')
	for j in range(n): # VIS
		for k in range(n): # RSC
			plot([phi[j],phi[k]], [0,1], 'o-', linewidth = stdp[j,i,k]*200.0, markersize = 0.5)	
	ax.set_xticklabels([])		

figure()
maps = makePlaceFields(center_pc, sigma, WINDOW_WIDTH, WINDOW_HEIGHT).sum(0)
imshow(maps, origin = 'upper right', extent = (0, WINDOW_WIDTH, 0, WINDOW_HEIGHT))
plot(agent.data['x'], agent.data['y'])

figure()
dtt = 100
phi_decoded = pd.DataFrame(index = np.arange(0, duration/ms, dtt))
for gr,nme in zip([cw_mon, ccw_mon, adn_mon, pos_mon], ['lmncw_', 'lmnccw_', 'adn_', 'pos_']):
	frate = pd.DataFrame(index = np.arange(0, duration/ms, dtt))
	spikes = gr.spike_trains()
	for neur in spikes.keys():
		f, bin_edges = np.histogram(spikes[neur]/ms, int(duration/ms/dtt), range = (0, duration/ms))
		frate[nme+str(neur)] = f/dtt
	phi_decoded[nme] = phi[np.argmax(frate.values, 1)]

plot(phi_decoded)
plot(agent.data['theta'], color = 'black', lw = 5)

figure()
subplot(211)
plot(angular_velocity)
subplot(212)
plot(CW_noise.values)
plot(CCW_noise.values)


show(block = False)