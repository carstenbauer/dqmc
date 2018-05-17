cd("..")
input = "tests/tests_linalg.in.xml"
include("live.jl")
cd("tests")


# -------------------------------
#		High temperature
# -------------------------------
mc.p.slices = 20
mc.p.delta_tau = 0.1
mc.p.beta = 2
init!(mc)
@assert mc.p.slices == 20


Ut, Dt, Vt, svst = calc_Bchain(mc, 1, mc.p.slices, 1);
Uv, Dv, Vdv, svsv = calc_Bchain_udv(mc, 1, mc.p.slices, 1);
B = Ut*spdiagm(Dt)*Vt;
# B = rand!(similar(mc.s.greens));
# Uv, Dv, Vdv = decompose_udv(B); # svd
# Ut, Dt, Vt = decompose_udt(B); # qr

# calculation of inverse
x = inv_udv(Uv, Dv, Vdv);
y = inv_udt(Ut, Dt, Vt);
compare(x,y) # max absdiff: 1.3e-11, max reldiff: 1.1e-10
compare(x,inv(B)) # max absdiff: 1.2e-09, max reldiff: 3.3e-09
compare(y,inv(B)) # max absdiff: 1.2e-09, max reldiff: 3.3e-09

# calculation of (1 + UDVd)^-1
x = inv_one_plus_udv(Uv,Dv,Vdv);
y = inv_one_plus_udv_scalettar(Uv,Dv,Vdv);
compare(x,y) # max absdiff: 3.0e-15, max reldiff: 6.8e-12
compare(x, inv(eye(B) + B)) # max absdiff: 8.0e-15, max reldiff: 1.0e-11
compare(y, inv(eye(B) + B)) # max absdiff: 6.5e-15, max reldiff: 1.5e-11


# -------------------------------
#		Low temperature
# -------------------------------
mc.p.slices = 400
mc.p.delta_tau = 0.1
mc.p.beta = 40
init!(mc)
@assert mc.p.slices == 400

Ut, Dt, Vt, svst = calc_Bchain(mc, 1, mc.p.slices, 1);
Uv, Dv, Vdv, svsv = calc_Bchain_udv(mc, 1, mc.p.slices, 1);
B = Ut*spdiagm(Dt)*Vt;

# calculation of inverse
x = inv_udv(Uv, Dv, Vdv);
y = inv_udt(Ut, Dt, Vt);
compare(x,y) # max absdiff: 1.1e+62, max reldiff: 5.2e+01
compare(x,inv(B)) # max absdiff: 1.7e+62, max reldiff: 2.0e+00
compare(y,inv(B)) # max absdiff: 1.2e+62, max reldiff: 2.0e+00

# calculation of (1 + UDVd)^-1
x = inv_one_plus_udv(Uv,Dv,Vdv);
y = inv_one_plus_udv_scalettar(Uv,Dv,Vdv);
compare(x,y) # max absdiff: 5.6e-01, max reldiff: 2.0e+00
compare(x, inv(eye(B) + B)) # max absdiff: 7.5e-13, max reldiff: 2.0e+00
compare(y, inv(eye(B) + B)) # max absdiff: 5.6e-01, max reldiff: 2.0e+00


eyeUv, eyeDv, eyeVdv = decompose_udv(eye(B));
T1, T2, T3 = inv_sum_udvs(eyeUv, eyeDv, eyeVdv, Uv, Dv, Vdv);
z = T1*spdiagm(T2)*T3;
compare(x,z)
compare(y,z)

# ----------------------------------------
#		High temperature, many slices
# ----------------------------------------
mc.p.slices = 200
mc.p.delta_tau = 0.01
mc.p.beta = 2
init!(mc)
@assert mc.p.slices == 200 && mc.p.delta_tau == 0.01

Ut, Dt, Vt, svst = calc_Bchain(mc, 1, mc.p.slices, 1);
Uv, Dv, Vdv, svsv = calc_Bchain_udv(mc, 1, mc.p.slices, 1);
B = Ut*spdiagm(Dt)*Vt;

# calculation of inverse
x = inv_udv(Uv, Dv, Vdv);
y = inv_udt(Ut, Dt, Vt);
compare(x,y) # max absdiff: 5.5e+22, max reldiff: 3.3e-02
compare(x,inv(B)) # max absdiff: 1.6e+29, max reldiff: 2.0e+00
compare(y,inv(B)) # max absdiff: 1.6e+29, max reldiff: 2.0e+00

# calculation of (1 + UDVd)^-1
x = inv_one_plus_udv(Uv,Dv,Vdv);
y = inv_one_plus_udv_scalettar(Uv,Dv,Vdv);
compare(x,y) # max absdiff: 6.3e-01, max reldiff: 1.3e+02
compare(x, inv(eye(B) + B)) # max absdiff: 2.8e-04, max reldiff: 1.3e+01
compare(y, inv(eye(B) + B)) # max absdiff: 6.3e-01, max reldiff: 1.5e+01




using PyPlot

figure()
plot(svst')
xlabel("imaginary time slices")
ylabel("QR scales")

figure()
plot(svsv')
xlabel("imaginary time slices")
ylabel("SVD scales")