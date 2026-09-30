import numpy as np, run_geometry_sweep as G, sata1d as S
def lvl(cfg,x,pk,case):
    t=(np.arange(cfg.Na)-cfg.Na//2)/cfg.prf
    return round(G.ambiguity_level(t,np.abs(x),pk,case.xs,cfg.system.vs,G.ambiguity_times(cfg)),1)
for sp,N,a in ((2600,5,20),(2600,5,0.1),(2600,3,5),(2600,2,5),(200,5,0.1),(200,2,5),(200,13,30),(200,13,1)):
    h=G.heights_from_slopes([a]*(N-1),200.0)
    c=G.Case("ramp",f"N{N}_a{a}","",h,float(sp),{})
    cfg,foc=G.simulate(c,G.BXT_MAX)
    cf0,f0=G.simulate(c,0.0)
    flat=f0["none"]; pk=np.abs(foc["ref"]).max()
    o={k:(lvl(cfg,v,pk,c),lvl(cfg,v-foc["ref"],pk,c),lvl(cfg,v-flat,pk,c)) for k,v in foc.items() if k!="ref"}
    print(sp,N,a,"flat vs ref:",lvl(cfg,flat,pk,c),lvl(cfg,flat-foc["ref"],pk,c),o)
