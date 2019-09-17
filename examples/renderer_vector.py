import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

def draw_dot(pos, size):
	plt.scatter(pos[0], pos[1], s=size)

def draw_line(pos_start, pos_end):
	plt.scatter(3, 9, s=1000)

def draw_rectangle(ax):
	p_fancy = mpatches.Rectangle([8,4.5], 1, 1, 45)	
	#t2 = mpl.transforms.Affine2D().rotate_deg(-45) + ax.transData
	#p_fancy.set_transform(t2)
	ax.add_patch(p_fancy)

def draw_fancy_rectangle(ax):
	p_fancy = mpatches.FancyBboxPatch([8,4.5], 1, 1)	
	#t2 = mpl.transforms.Affine2D().rotate_deg(-45) + ax.transData
	#p_fancy.set_transform(t2)
	ax.add_patch(p_fancy)

def main():
  	fig = plt.figure(figsize=(16,9))
  	#plt.axis('off')
#  	plt.gca().set_aspect('equal', adjustable='box')
  	ax = plt.gca()
  	ax.set_xlim([0,16])
  	ax.set_ylim([0,9])

	draw_rectangle(ax)

if __name__ == '__main__':
  main()
  plt.show()
