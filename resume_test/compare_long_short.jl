using MonteCarloObservable, Helpers

c = ts_flat("resumelong.out.h5", "obs/configurations");
cc = ts_flat("resumeshort.out.h5", "obs/configurations");
# compare(c[:,:,:,1:100], cc)
@assert isapprox(c[:,:,:,1:100], cc)

g = ts_flat("resumelong.out.h5", "obs/greens");
gg = ts_flat("resumeshort.out.h5", "obs/greens");
# compare(g[:,:,1:100], gg)
@assert isapprox(g[:,:,:,1:100], gg)