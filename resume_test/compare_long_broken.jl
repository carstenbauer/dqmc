using MonteCarloObservable, Helpers

c = ts_flat("resumelong.out.h5", "obs/configurations");
cc = ts_flat("resumebroken.out.h5", "obs/configurations");
# compare(c, cc)
@assert isapprox(c,cc)

g = ts_flat("resumelong.out.h5", "obs/greens");
gg = ts_flat("resumebroken.out.h5", "obs/greens");
# compare(g, gg)
@assert isapprox(g,gg)