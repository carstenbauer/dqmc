import os, sys, glob

dir_pat = "L_{L}/sdwO3_L_{L}_B_{beta}_dt_{dt}_{seednr}"
sh_pat = "sdwO3_L_{L}_B_{beta}_dt_{dt}_{seednr}.sh"
xml_pat = "sdwO3_L_{L}_B_{beta}_dt_{dt}_{seednr}.task{j}.in.xml"
hdf_pat = "sdwO3_L_{L}_B_{beta}_dt_{dt}_{seednr}.task{j}.out.h5"

for L in [6]:
	for beta in [20]:
		for dt in [0.1]:
			for (k, seed) in enumerate([55796]):
				seednr = k+1
				d = dir_pat.format(L=L, beta=beta, dt=dt, seednr=seednr)

				if (os.path.isdir(d)):
					os.chdir(d)
					submit = []
            
				input_files = glob.glob("*.in.xml")
				for j in range(1,len(input_files)+1):
					xml = xml_pat.format(L=L, beta=beta, dt=dt, seednr=seednr, j=j)
					hdf = hdf_pat.format(L=L, beta=beta, dt=dt, seednr=seednr, j=j)
					sh =  sh_pat.format(L=L, beta=beta, dt=dt, seednr=seednr, j=j)
					if os.path.isfile(xml) and not os.path.isfile(hdf):
						submit.append(j)

				if len(submit) > 0:
					ids = ",".join([str(s) for s in submit])
					print("sbatch --array={ids} {sh}".format(ids=ids, sh=sh))
					os.system("sbatch --array={ids} {sh}".format(ids=ids, sh=sh))
				os.chdir("../..")
        
            
