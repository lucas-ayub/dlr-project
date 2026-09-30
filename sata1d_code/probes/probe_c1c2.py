import numpy as np, sata1d as S
from sata1d.reconstruction import GetCoeffNu
WL=0.25
print("h [m] | per channel: dC0 [deg], residual phase spread over the band from dC1/dDt/dC2 [deg]")
for h in (100,500,1000,3000,10000,37000):
    cfg=S.make_config([(0.0,h)]); tr=S.build_platform_tracks(cfg); p=cfg.scene.points[1]
    f=np.linspace(-cfg.abw/2,cfg.abw/2,401); line=[]
    for i in range(cfg.Nrx):
        com=(tr.ptx,tr.prx[i],tr.vtx,tr.vrx[i],tr.ptx,tr.vtx,cfg.prf,WL,cfg.ta,cfg.sq_tx,cfg.sq_rx[i],cfg.theta_tx,cfg.theta_rx[i])
        a=np.array(GetCoeffNu(p,*com)); b=np.array(GetCoeffNu(cfg.scene.ptg,*com)); d=a-b
        ph=2*np.pi*((-d[1]+d[3])*f+d[2]*WL*f**2); ph-=ph.mean()
        line.append(f"ch{i}: {np.degrees(2*np.pi*d[0]/WL):9.1f} / {np.degrees(np.ptp(ph)):.3f}")
    print(h," | ".join(line))
