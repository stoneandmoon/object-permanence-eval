"""Current V2 formulation: evidence reliability, agreement, and persistence."""
import numpy as np
from .temporal_utils import clip01,rolling_median,robust_zscore,consecutive_confirmation,asymmetric_ema,reliability_gate,interpolate_short_invalid_gaps
def _base(x,v):
 q=np.asarray(x,float)[v&np.isfinite(x)][:20];return (float(np.nanmedian(q)),float(np.nanmedian(np.abs(q-np.nanmedian(q))))) if len(q) else (1.,.05)
def _pen(x,v):
 b,m=_base(x,v);return clip01((np.abs(np.asarray(x,float)-b)/max(1.4826*m,.05*abs(b),.02)-2)/4)
def _agree(*x):
 q=np.sort(np.nan_to_num(np.stack(x)),axis=0);return clip01(.60*q[-2]+.40*q[-1])
def compute(frame,fps=16.):
 p=np.asarray(frame['mask_present'],float);a=np.asarray(frame['mask_area_ratio'],float);ident=np.asarray(frame['identity_similarity'],float);iv=np.asarray(frame['identity_valid'],float);depth=np.asarray(frame['depth_valid_ratio'],float);iou=np.asarray(frame['prev_mask_iou'],float);tr,ir,dr,rel=reliability_gate(p,a,iv,depth,iou);valid=(tr>.6)&(ir>.5)
 ep1=(1-p)*.60;ep2=np.where(iv>.5,clip01((.65-ident)/.45),0);ep3=(1-depth)*.35;es=_agree(ep1,ep2,ep3);ec=consecutive_confirmation(es);ex=asymmetric_ema(1-np.where(ec,es,np.minimum(es,.12)),rel)
 af=rolling_median(interpolate_short_invalid_gaps(a));xf=rolling_median(interpolate_short_invalid_gaps(frame['aspect_ratio']));cf=rolling_median(interpolate_short_invalid_gaps(frame['compactness']));sa,sx,sc=_pen(np.log(np.maximum(af,1e-8)),valid),_pen(np.log(np.maximum(xf,1e-8)),valid),_pen(cf,valid);ss=np.where(rel>.45,_agree(sa,sx,sc),0);sok=consecutive_confirmation(ss);sh=asymmetric_ema(1-np.where(sok,ss,np.minimum(ss,.10)),rel)
 x=rolling_median(interpolate_short_invalid_gaps(frame['centroid_x']));y=rolling_median(interpolate_short_invalid_gaps(frame['centroid_y']));v=np.hypot(np.diff(x,prepend=x[0]),np.diff(y,prepend=y[0]))*fps;acc=np.abs(np.diff(v,prepend=v[0]))*fps;flow=rolling_median(interpolate_short_invalid_gaps(frame['flow_magnitude']));mr=rolling_median(interpolate_short_invalid_gaps(frame['motion_residual']));rr=rolling_median(interpolate_short_invalid_gaps(frame['trajectory_residual']));mj,ma,mf,mt=clip01((robust_zscore(v)-2)/4),clip01((robust_zscore(acc)-2)/4),clip01((robust_zscore(flow)-2)/4),clip01((robust_zscore(mr+rr)-2)/4);ms=_agree(mj,ma,mf,mt);mok=consecutive_confirmation(ms);mo=asymmetric_ema(1-np.where(mok,ms,np.minimum(ms,.10)),np.maximum(rel,.45))
 return dict(existence=ex,shape=sh,motion=mo,reliability=rel,existence_severity=es,shape_severity=ss,motion_severity=ms,existence_confirmed=ec,shape_confirmed=sok,motion_confirmed=mok)
