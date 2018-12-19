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

\title{IR Graphs}

\begin{document}

\maketitle

\section{Scalar IR}

\tikz []
%\tikz
\graph [layered layout, components go right top aligned, nodes=draw, edges=rounded corners]
{
%    first root -> {1 -> {2, 3, 7} -> {4, 5}, 6 }, 4 -- 5, 7 -- 4;

'''

tail = r'''
};

\section{Vector IR}

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

os.system("cd {} && lualatex main.tex".format(folder_name))
os.system("cp {}/main.pdf {}".format(folder_name, sys.argv[1]))
os.system("rm -rf {}".format(folder_name))
