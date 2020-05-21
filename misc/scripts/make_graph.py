import os
import random
import sys

head = r'''
\documentclass{article}

\usepackage{tikz}
\usepackage{luatex85}
\usetikzlibrary{arrows.meta}
\usetikzlibrary{angles}
\usetikzlibrary{trees}
\usetikzlibrary{arrows,decorations.pathmorphing,backgrounds,positioning,fit,petri}
\usetikzlibrary{graphs}
\usetikzlibrary{bending}
\usetikzlibrary{automata}
\usetikzlibrary{graphdrawing,graphs}
\usegdlibrary{layered}
\usetikzlibrary{shapes.multipart}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}

\title{IR Visualization}

\begin{document}

\maketitle

\begin{center}
\scalebox{0.2}{
\tikz []
%\tikz
\graph [layered layout, components go right top aligned, nodes=draw, edges=rounded corners]
{

'''

tail = r'''
};}
\end{center}

\end{document}

'''

# parameters: filename, graph

assert len(sys.argv) == 3
folder_name = "/tmp/_graph_cache_{:04d}".format(random.randint(0, 10000))
os.mkdir(folder_name)

with open(os.path.join(folder_name, "main.tex"), 'w') as f:
    f.write(head)
    f.write(sys.argv[2])
    f.write(tail)

os.system("cd {} && lualatex main.tex > /dev/null".format(folder_name))
os.system("cp {}/main.pdf {}".format(folder_name, sys.argv[1]))
os.system("rm -rf {}".format(folder_name))
