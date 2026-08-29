"""Realtime curve review renderer using existing video frames and masks only."""
from __future__ import annotations
from pathlib import Path
import subprocess,cv2,numpy as np
def _encode(temp,out):
 r=subprocess.run(['ffmpeg','-y','-v','error','-i',str(temp),'-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(out)],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 temp.unlink()
def render(video,mask_dir,target,fps,curves,out,name):
 """Render existing contour plus progressive curves; no tracking inference."""
 cap=cv2.VideoCapture(str(video));frames=[]
 while True:
  ok,f=cap.read()
  if not ok:break
  frames.append(f)
 cap.release();n=len(frames)
 if not n or any(len(x)!=n for x in curves.values()):raise ValueError('video/curve length mismatch')
 h,w=frames[0].shape[:2];temp=Path(out)/(name+'.tmp.mp4');wr=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h));colors=[(210,100,25),(0,120,230),(50,130,45)]
 for i,f in enumerate(frames):
  m=cv2.imread(str(Path(mask_dir)/f'{i:06d}.png'),0)
  if m is None:raise FileNotFoundError(Path(mask_dir)/f'{i:06d}.png')
  left=f.copy();cs,_=cv2.findContours((m>0).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(left,cs,-1,(0,255,0),2);cv2.putText(left,f'{target} | frame {i}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2)
  right=np.full((h,w,3),250,np.uint8);lm,top,pw,ph=55,50,w-75,h-100;cv2.rectangle(right,(lm,top),(lm+pw,top+ph),(130,130,130),1)
  for j,(key,values) in enumerate(curves.items()):
   pts=np.array([(lm+int(k*pw/max(1,n-1)),top+int((1-np.clip(values[k],0,1))*ph)) for k in range(i+1)],np.int32)
   if len(pts)>1:cv2.polylines(right,[pts],False,colors[j],2)
   cv2.putText(right,f'{key}: {values[i]:.3f}',(lm,top+ph+18+j*17),cv2.FONT_HERSHEY_SIMPLEX,.38,colors[j],1)
  xx=lm+int(i*pw/max(1,n-1));cv2.line(right,(xx,top),(xx,top+ph),(0,0,0),1);wr.write(np.hstack([left,right]))
 wr.release();_encode(temp,Path(out)/(name+'.mp4'))
