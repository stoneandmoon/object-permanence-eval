"""Shared V2 temporal primitives; all windows are deliberately short."""
import numpy as np
import pandas as pd
def clip01(x): return np.clip(np.asarray(x,float),0.,1.)
def rolling_median(x, window=5): return pd.Series(x,dtype=float).rolling(window,center=True,min_periods=1).median().to_numpy()
def rolling_mad(x, window=5): return pd.Series(x,dtype=float).rolling(window,center=True,min_periods=1).apply(lambda a:np.nanmedian(np.abs(a-np.nanmedian(a))),raw=True).to_numpy()
def robust_zscore(x, window=5):
 x=np.asarray(x,float);return np.abs(x-rolling_median(x,window))/(1.4826*rolling_mad(x,window)+1e-5)
def interpolate_short_invalid_gaps(x,max_gap=2):
 x=np.asarray(x,float).copy();good=np.where(np.isfinite(x))[0]
 if len(good):
  for i in range(len(x)):
   if not np.isfinite(x[i]):
    lo=good[good<i];hi=good[good>i]
    if len(lo) and len(hi) and hi[0]-lo[-1]-1<=max_gap:x[i]=np.interp(i,[lo[-1],hi[0]],[x[lo[-1]],x[hi[0]]])
 return x
def consecutive_confirmation(severity,enter=.55,exit=.30,frames=3,extreme=.88):
 s=clip01(severity);o=np.zeros(len(s),bool);state=False;run=0
 for i,v in enumerate(s):
  run=run+1 if v>=enter else 0
  if not state and (run>=frames or v>=extreme):state=True
  elif state and v<=exit:state=False
  o[i]=state
 return o
def asymmetric_ema(target,reliability,attack=.35,recovery=.10,initial=1.):
 last=initial;o=[]
 for t,r in zip(clip01(target),clip01(reliability)):
  if r>=.35:last+=(attack if t<last else recovery)*(t-last)
  o.append(last)
 return clip01(o)
def reliability_gate(mask_present,area,identity_valid,depth_valid,iou):
 a=np.asarray(area,float);ref=np.nanmedian(a[(a>0)&np.isfinite(a)][:20]) or 1.;track=clip01(.65*np.asarray(mask_present,float)+.20*clip01(a/(.25*ref))+.15*np.nan_to_num(iou,nan=.7));identity=np.asarray(identity_valid,float);depth=clip01(depth_valid);return track,identity,depth,clip01(.50*track+.30*identity+.20*depth)
