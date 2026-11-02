import torch
from dpvo.net import altcorr,coords_grid_with_index,DIM

def masked_forward(self,images,patches_per_image=80,disps=None,centroid_sel_strat='RANDOM',return_color=False):
    fmap=self.fnet(images)/4.;imap=self.inet(images)/4.;b,n,c,h,w=fmap.shape;P=self.patch_size
    mask=torch.nn.functional.interpolate(self.current_static.float()[None,None],size=(h,w),mode='nearest')[0,0]>0
    mask[:1]=False;mask[-1:]=False;mask[:,:1]=False;mask[:,-1:]=False
    valid=torch.nonzero(mask,as_tuple=False);self.mask_counts.append(int(len(valid)))
    if len(valid)>=8:
        picked=valid[torch.randint(len(valid),(n,patches_per_image),device='cuda')];y=picked[:,:,0];x=picked[:,:,1]
    else:
        x=torch.randint(1,w-1,size=[n,patches_per_image],device='cuda');y=torch.randint(1,h-1,size=[n,patches_per_image],device='cuda')
    coords=torch.stack([x,y],dim=-1).float();imap=altcorr.patchify(imap[0],coords,0).view(b,-1,DIM,1,1);gmap=altcorr.patchify(fmap[0],coords,P//2).view(b,-1,128,P,P)
    if return_color:clr=altcorr.patchify(images[0],4*(coords+.5),0).view(b,-1,3)
    if disps is None:disps=torch.ones(b,n,h,w,device='cuda')
    grid,_=coords_grid_with_index(disps,device=fmap.device);patches=altcorr.patchify(grid[0],coords,P//2).view(b,-1,3,P,P);index=torch.arange(n,device='cuda').view(n,1).repeat(1,patches_per_image).reshape(-1)
    return (fmap,gmap,imap,patches,index,clr) if return_color else (fmap,gmap,imap,patches,index)
