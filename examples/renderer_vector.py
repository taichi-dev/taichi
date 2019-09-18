import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.animation as animation

class VectorRenderer:
	def __init__(self):
		self.canvas_scale = [9.6,7.2]
		plt.ion()
		self.fig = plt.figure(figsize=(self.canvas_scale[0],self.canvas_scale[1]))
		self.ax = self.fig.add_subplot(111)
		self.fig.show()

	def draw_dot(self, pos, size):
		plt.scatter(self.canvas_scale[0] * pos[0], self.canvas_scale[1] * pos[1], s=size)

	def draw_line(self, pos_start, pos_end, dashed = False):
		plt.plot([self.canvas_scale[0] * pos_start[0], self.canvas_scale[0] * pos_end[0]], [self.canvas_scale[1] * pos_start[1], self.canvas_scale[1] * pos_end[1]], linewidth=3)

	def draw_rectangle(self, pos, angle = 0):
		p_fancy = mpatches.Rectangle([pos[0],pos[1]], 1, 1, angle)	
		self.ax.add_patch(p_fancy)

	def draw_polygon(self, pos):
		p_fancy = mpatches.Polygon(pos)	
		self.ax.add_patch(p_fancy)

	def draw_fancy_rectangle(self,ax):
		p_fancy = mpatches.FancyBboxPatch([8,4.5], 1, 1)
		ax.add_patch(p_fancy)

	def build_axis(self):
		self.ax.axis('off')
		self.ax.set_xlim([0,self.canvas_scale[0]])
		self.ax.set_ylim([0,self.canvas_scale[1]])

	def clean_frame(self):
		self.fig.canvas.flush_events()
		#time.sleep(0.000001)
		plt.cla()