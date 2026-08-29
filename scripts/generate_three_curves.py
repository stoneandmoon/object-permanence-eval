#!/usr/bin/env python3
"""Default V2 three-curve generator; reads evidence only, never runs models."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from modules.evidence import load_evidence_csv
from modules.curves_v2.logic import compute
from modules.curves_v2.realtime import render as render_realtime
def canonical(rows):
 def col(name,default=np.nan):return pd.to_numeric(rows[name],errors='coerce').to_numpy() if name in rows else np.full(len(rows),default)
 return {'mask_present':rows.mask_exists.astype(float).to_numpy(),'mask_area_ratio':col('mask_area'),'identity_similarity':col('identity_similarity'),'identity_valid':np.isfinite(col('identity_similarity')).astype(float),'depth_valid_ratio':col('depth_valid_ratio',0),'prev_mask_iou':col('previous_mask_iou'),'aspect_ratio':col('aspect_ratio'),'compactness':col('compactness'),'centroid_x':col('centroid_x'),'centroid_y':col('centroid_y'),'flow_magnitude':col('local_motion_after_camera_compensation'),'motion_residual':col('motion_residual',0),'trajectory_residual':col('trajectory_residual')}
def plot(out,three):
 fig,ax=plt.subplots(figsize=(12,4.5))
 for k,n in [('object_existence','Object Existence'),('shape_normality','Shape Normality'),('motion_smoothness','Motion Smoothness')]:ax.plot(three.frame,three[k],label=n)
 ax.set(xlabel='Frame',ylabel='Score',title='Object Permanence Curves (V2)',ylim=(0,1));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'three_curves.png',dpi=160);plt.close(fig)
def main():
 p=argparse.ArgumentParser(description='Generate temporally robust Three Curves V2 from existing multimodal evidence')
 p.add_argument('--evidence-csv');p.add_argument('--tracking-dir');p.add_argument('--video',help='optional provenance only; no inference');p.add_argument('--mask-dir',help='reserved for the project realtime renderer');p.add_argument('--target',default='target');p.add_argument('--task');p.add_argument('--output',required=True);a=p.parse_args()
 if bool(a.evidence_csv)==bool(a.tracking_dir):p.error('provide exactly one of --evidence-csv or --tracking-dir')
 source=Path(a.evidence_csv) if a.evidence_csv else Path(a.tracking_dir)/'evidence/per_instance_frame_evidence.csv';e=load_evidence_csv(source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);all=[]
 for iid,rows in e.groupby('instance_id',sort=True):
  rows=rows.sort_values('frame_idx').reset_index(drop=True);r=compute(canonical(rows));all.append(pd.DataFrame({'frame':rows.frame_idx,'time_sec':rows.timestamp,'instance_id':iid,'object_existence':r['existence'],'shape_normality':r['shape'],'motion_smoothness':r['motion'],'overall_reliability':r['reliability'],'existence_severity':r['existence_severity'],'shape_severity':r['shape_severity'],'motion_severity':r['motion_severity']}))
 allc=pd.concat(all);three=allc.groupby(['frame','time_sec'],as_index=False)[['object_existence','shape_normality','motion_smoothness']].mean();allc.to_csv(out/'per_instance_three_curves.csv',index=False);three.to_csv(out/'three_curves.csv',index=False)
 for k,n in [('object_existence','object_existence_curve.csv'),('shape_normality','shape_normality_curve.csv'),('motion_smoothness','motion_smoothness_curve.csv')]:three[['frame','time_sec',k]].to_csv(out/n,index=False)
 plot(out,three)
 if a.video or a.mask_dir:
  if not (a.video and a.mask_dir):p.error('--video and --mask-dir must be supplied together for realtime rendering')
  preview=out/'preview';preview.mkdir(exist_ok=True);sets=[('realtime_object_existence',{'Existence':three.object_existence.to_numpy()}),('realtime_shape_normality',{'Shape':three.shape_normality.to_numpy()}),('realtime_motion_smoothness',{'Motion':three.motion_smoothness.to_numpy()}),('realtime_three_curves',{'Existence':three.object_existence.to_numpy(),'Shape':three.shape_normality.to_numpy(),'Motion':three.motion_smoothness.to_numpy()})]
  for name,values in sets:render_realtime(a.video,a.mask_dir,a.target,16,values,preview,name)
 (out/'curve_generation.json').write_text(json.dumps({'curve_version':'v2','model_inference_run':False,'video_encoding':'realtime MP4: libx264/yuv420p/faststart'},indent=2)+'\n')
if __name__=='__main__':main()
