# 2x short means one short run, then double measurements value 
# in .in.xml and rename resumeshort.out.h5 -> resumeshort.out.h5.running
# and run a second time

using MonteCarloObservable, Helpers

c = ts_flat("resumelong.out.h5", "obs/configurations");
cc = ts_flat("resumeshort.out.h5", "obs/configurations");
# compare(c, cc)
@assert isapprox(c,cc)

g = ts_flat("resumelong.out.h5", "obs/greens");
gg = ts_flat("resumeshort.out.h5", "obs/greens");
# compare(g, gg)
@assert isapprox(g,gg)