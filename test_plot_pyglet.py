import pyglet, io
import numpy as np
from time import time
import sys
import matplotlib
matplotlib.use("GTKAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
dpi_res = 60
fig = Figure((100 / dpi_res, 100 / dpi_res), dpi = dpi_res)
X = np.linspace(-6, 6, 1024)
Y = np.sinc(X)
ax = fig.add_subplot(111)
line, = ax.plot(X, Y, lw= 2, color = 'k')
ax.set_ylim(0, 1)
ax.set_autoscale_on(False)
w, h = fig.get_size_inches()
dpi_res = fig.get_dpi()
w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))


canvas = FigureCanvasAgg(fig)
canvas.draw()
s = canvas.tostring_rgb()
image = pyglet.image.ImageData(w, h, 'RGB', data = s, pitch = -3*w)
background = fig.canvas.copy_from_bbox(ax.bbox)


def update(dt):    
    print(dt)    
    fig.canvas.restore_region(background)
    line.set_ydata(np.random.rand(len(X)))    
    ax.draw_artist(line)
    canvas.blit(ax.bbox)
    image.set_data(data = canvas.tostring_rgb(), format = 'RGB', pitch = -3*w)
    # window.clear()
    image.blit(700, 500)


@window.event
def on_draw():
    # window.clear()
    # image.blit(0,0)
    pass

pyglet.clock.schedule_interval(update, 1/100.)
pyglet.app.run()
