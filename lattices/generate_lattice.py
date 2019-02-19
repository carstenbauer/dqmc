import sys
from subprocess import call

try:
	l = str(sys.argv[1])
	L = str(sys.argv[2])
	W = str(sys.argv[3])
except:
	print "python generate_lattice.py <lattice_name> <L> <W>"
	exit()

l_alps = {"honeycomb": "honeycomb lattice","square": "square lattice", "chain": "chain lattice"}[l]

filestr = "LATTICE=\""+l_alps+"\"\nL="+L+"\nW="+W
#print filestr

f = open("lattice.in","w")
f.write(filestr)
f.close()

f = open(l+"_"+"L_"+L+"_W_"+W+".xml","w")
call(["printgraph", "lattice.in"], stdout=f)
f.close()
call(["rm", "lattice.in"])


