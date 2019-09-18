import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.animation as animation



class VectorRenderer:
	def __init__(self):
		plt.ion()
		self.fig = plt.figure(figsize=(16,9))
		self.ax = self.fig.add_subplot(111)
		self.fig.show()
		#self.camera = Camera(self.fig)
	  	#plt.axis('off')
		#self.ax = self.fig.add_subplot(1,1,1)

		#plt.ion()

		#self.draw_line([0,0],[10,10])

	def draw_dot(self, pos, size):
		plt.scatter(pos[0], pos[1], s=size)

	def draw_line(self, pos_start, pos_end, dashed = False):
		plt.plot([pos_start[0],pos_end[0]], [pos_start[1],pos_end[1]], linewidth=3)

	def draw_rectangle(self, pos, angle = 0):
		p_fancy = mpatches.Rectangle([pos[0],pos[1]], 1, 1, angle)	
		self.ax.add_patch(p_fancy)

	def draw_fancy_rectangle(self,ax):
		p_fancy = mpatches.FancyBboxPatch([8,4.5], 1, 1)	
		#t2 = mpl.transforms.Affine2D().rotate_deg(-45) + ax.transData
		#p_fancy.set_transform(t2)
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
