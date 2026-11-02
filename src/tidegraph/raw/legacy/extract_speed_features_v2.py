import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

def features(prev,rgb,dt,cal):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    flow=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)/(max(dt,1e-3))
    # Rates in calibrated normalized image coordinates (approximately rad/s).
    flow[:,:,0]/=cal['fx']*160/cal['width'];flow[:,:,1]/=cal['fy']*90/cal['height']
    row=np.median(gray[:,8:-8],axis=1);edge=np.abs(np.gradient(gaussian_filter1d(row.astype(float),1)))
    hy=36+np.argmax(edge[36:77]);values=[hy/90,cal['fx']/cal['width'],cal['fy']/cal['height']]
    for iy in range(3):
        for ix in range(4):
            ys=slice(iy*30,(iy+1)*30);xs=slice(ix*40,(ix+1)*40)
            colors=rgb[ys,xs].reshape(-1,3).astype(float)/255
            f=flow[ys,xs].reshape(-1,2)
            values.extend(colors.mean(0));values.extend(colors.std(0))
            values.extend(np.quantile(f,[.1,.5,.9],axis=0).ravel())
            values.extend(np.std(f,axis=0))
            values.append(float(np.mean(np.abs(cv2.Laplacian(gray[ys,xs],cv2.CV_32F))))/255)
    return np.array(values,dtype=np.float32)
