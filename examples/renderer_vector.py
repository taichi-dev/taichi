import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.animation as animation

class VectorRenderer:
	def __init__(self):
		self.canvas_scale = [16,9]
		plt.ion()
		self.fig = plt.figure(figsize=(self.canvas_scale[0],self.canvas_scale[1]))
		self.ax = self.fig.add_subplot(111)
		self.fig.show()

	def draw_dot(self, pos, size):
		plt.scatter(pos[0], pos[1], s=size)

	def draw_line(self, pos_start, pos_end, dashed = False):
		plt.plot([self.canvas_scale[0] * pos_start[0], self.canvas_scale[0] * pos_end[0]], [self.canvas_scale[1] * pos_start[1], self.canvas_scale[1] * pos_end[1]]*self.canvas_scale[1], linewidth=30)

	def draw_rectangle(self, pos, angle = 0):
		p_fancy = mpatches.Rectangle([pos[0],pos[1]], 1, 1, angle)	
		self.ax.add_patch(p_fancy)

	def draw_fancy_rectangle(self,ax):
		p_fancy = mpatches.FancyBboxPatch([8,4.5], 1, 1)
		ax.add_patch(p_fancy)

	def clean(self):
		self.ax.set_xlim([0,16])
		self.ax.set_ylim([0,9])
		self.ax.clear()
		#self.ax = plt.gca()
		#self.ax.set_xlim([0,16])
		#self.ax.set_ylim([0,9])

#	def main():


#if __name__ == '__main__':
#	main()
#	plt.show()
