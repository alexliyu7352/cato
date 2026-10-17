"""Render real 3D offscreen VTK RGB, semantic, depth, normals and motion.

Only a geometric pipeline-smoke fixture; not an artistic quality benchmark.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from .npr import stylize,edges

PALETTE={1:(.92,.68,.53),2:(.26,.57,.32),3:(.79,.62,.44),4:(.57,.70,.44),5:(.51,.56,.61),6:(.32,.59,.75)}

def mesh(kind):
    if kind=='cube':return vtk.vtkCubeSource()
    if kind=='cone':p=vtk.vtkConeSource();p.SetResolution(12);p.SetDirection(0,0,1);return p
    p=vtk.vtkSphereSource();p.SetThetaResolution(18);p.SetPhiResolution(18);return p

def add(renderer,registry,category,kind,loc,scale,color=None,moving=False):
    s=mesh(kind);s.Update();mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(s.GetOutputPort())
    a=vtk.vtkActor();a.SetMapper(mapper);a.SetPosition(*loc);a.SetScale(*scale)
    pr=a.GetProperty();pr.SetColor(*(color or PALETTE[category]));pr.SetAmbient(.35);pr.SetDiffuse(.65)
    renderer.AddActor(a);registry.append((a,category,tuple(loc),moving,pr.GetColor()))

def build_scene(kind,w,h):
    r=vtk.vtkRenderer();r.SetBackground(.58,.78,.94);r.SetBackground2(.94,.97,.99);r.GradientBackgroundOn()
    win=vtk.vtkRenderWindow();win.SetSize(w,h);win.SetOffScreenRendering(1);win.SetMultiSamples(0);win.AddRenderer(r)
    reg=[]
    add(r,reg,4,'cube',(0,0,-.2),(28,28,.4))
    add(r,reg,4,'cube',(0,-1,.015),(2.7,17,.03),(.72,.60,.46))
    for x,y in [(-6,2),(-4,4),(6,-2),(7,3),(-7,-3)]:
        add(r,reg,2,'cube',(x,y,.9),(.28,.28,1.8),(.48,.30,.21))
        add(r,reg,2,'sphere',(x,y,2.2),(1.12,.96,1.16),(.30,.61,.31))
        add(r,reg,2,'sphere',(x+.32,y,2.68),(.80,.84,.70),(.25,.49,.30))
    for x,y,s in [(4,2,1),(-3,3,.8)]:
        add(r,reg,3,'cube',(x,y,1.1*s),(3*s,2.3*s,2.2*s))
        add(r,reg,3,'cone',(x,y,2.9*s),(3.0*s,3*s,2.0*s),(.55,.30,.30))
        add(r,reg,3,'cube',(x,y-1.2*s,.72*s),(.68*s,.1*s,1.42*s),(.40,.27,.21))
    for x,y in [(2,-3),(-4,-2),(5,-5)]:add(r,reg,5,'sphere',(x,y,.25),(.50,.42,.43))
    # Geometric puppet intentionally not an artist-built character.
    for shape,loc,scale,color in [
        ('cone',(-.6,-2,1),(.7,.7,1.65),(.75,.29,.31)),
        ('sphere',(-.6,-2,2.18),(.45,.4,.52),(.95,.73,.60)),
        ('sphere',(-.6,-1.77,2.47),(.48,.35,.28),(.27,.20,.18)),
        ('sphere',(-.8,-2.34,2.22),(.06,.05,.07),(.1,.1,.12)),
        ('sphere',(-.4,-2.34,2.22),(.06,.05,.07),(.1,.1,.12)),
        ('cube',(-.9,-2,.26),(.17,.17,.52),(.32,.22,.18)),
        ('cube',(-.3,-2,.26),(.17,.17,.52),(.32,.22,.18))]:
        add(r,reg,1,shape,loc,scale,color,True)
    cam=r.GetActiveCamera()
    if kind=='portrait':cam.SetPosition(1.1,-7,3.6);cam.SetFocalPoint(-.6,-1.5,1.9);cam.SetViewAngle(36)
    elif kind=='forest':cam.SetPosition(-8,-13,8);cam.SetFocalPoint(-3,0,1.3);cam.SetViewAngle(48)
    else:cam.SetPosition(9,-15,8);cam.SetFocalPoint(0,0,1.4);cam.SetViewAngle(45)
    cam.SetViewUp(0,0,1);r.ResetCameraClippingRange()
    return r,win,reg

def read(window,depth=False):
    x=vtk.vtkWindowToImageFilter();x.SetInput(window);x.ReadFrontBufferOff()
    if depth:x.SetInputBufferTypeToZBuffer()
    else:x.SetInputBufferTypeToRGB()
    x.Update();out=x.GetOutput();w,h,_=out.GetDimensions()
    arr=vtk_to_numpy(out.GetPointData().GetScalars())
    return arr.reshape(h,w) [::-1].copy() if depth else arr.reshape(h,w,3)[::-1].copy()

def matrix(r):
    w,h=r.GetRenderWindow().GetSize()
    m=r.GetActiveCamera().GetCompositeProjectionTransformMatrix(w/h,0,1)
    return np.array([[m.GetElement(i,j) for j in range(4)] for i in range(4)])

def reconstruct(z,proj):
    h,w=z.shape;y,x=np.mgrid[:h,:w]
    ndc=np.stack([2*(x+.5)/w-1,1-2*(y+.5)/h,z,np.ones_like(z)],-1)
    pos=ndc.reshape(-1,4)@np.linalg.inv(proj).T
    return (pos[:,:3]/pos[:,3:4]).reshape(h,w,3)

def geometry(z,proj,semantic,frame):
    xyz=reconstruct(z,proj)
    normals=np.cross(np.gradient(xyz,axis=1),np.gradient(xyz,axis=0))
    normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-6)
    normals[z>.999]=0
    # Fixed camera: inverse displacement for rigid character translation only.
    old=xyz.copy();old[semantic==1,0]-=.08
    pts=np.concatenate([old,np.ones((*z.shape,1))],axis=-1).reshape(-1,4)
    clip=pts@proj.T;ndc=clip[:,:3]/np.maximum(np.abs(clip[:,3:4]),1e-8)
    h,w=z.shape;y,x=np.mgrid[:h,:w]
    flow=np.stack([((ndc[:,0]+1)*w*.5-.5).reshape(h,w)-x,((1-ndc[:,1])*h*.5-.5).reshape(h,w)-y],-1)
    valid=(z<.999)&(frame>0)&(np.abs(flow).max(axis=-1)<w/2)
    flow[~valid]=0
    return normals.astype('float32'),flow.astype('float32'),valid.astype('uint8')

def render(output,kind='village',frames=4,width=320,height=192):
    output=Path(output);r,win,reg=build_scene(kind,width,height);items=[]
    for t in range(frames):
        for a,category,pos,moving,color in reg:
            if moving:a.SetPosition(pos[0]+.08*t,pos[1],pos[2])
            a.GetProperty().SetColor(*color);a.GetProperty().LightingOn()
        r.SetBackground(.58,.78,.94);r.SetBackground2(.94,.97,.99);r.GradientBackgroundOn()
        win.Render();rgb=read(win);depth=read(win,True);proj=matrix(r)
        r.GradientBackgroundOff();r.SetBackground(0,0,0)
        for a,category,pos,moving,color in reg:
            a.GetProperty().LightingOff();a.GetProperty().SetColor(category/6.,0,0)
        win.Render();code=read(win)[:,:,0].astype('float32')
        sem=np.argmin(np.abs(code[:,:,None]-np.arange(7)[None,None,:]*(255/6.)),axis=-1).astype('uint8')
        normals,motion,valid=geometry(depth,proj,sem,t)
        label=stylize(rgb,sem,depth);line=edges(rgb,sem,depth)
        folder=output/f'{kind}_{t:04d}';folder.mkdir(parents=True,exist_ok=True)
        for name,img in [('source',rgb),('target',label),('semantic',sem),('line',line),('motion_valid',valid*255)]:
            Image.fromarray(img).save(folder/f'{name}.png')
        for name,x in [('depth',depth),('normal',normals),('motion_to_prev',motion)]:np.save(folder/f'{name}.npy',x.astype('float32'))
        rel=folder.relative_to(output)
        item=dict(scene=kind,frame=t,split='train' if t<frames-1 else 'val',
          target_origin='synthetic_npr_smoke_only',target_license='self_generated_smoke_fixture')
        for name in ['source','target','semantic','line','motion_valid']:item[name]=str(rel/f'{name}.png')
        for name in ['depth','normal','motion_to_prev']:item[name]=str(rel/f'{name}.npy')
        items.append(item)
    win.Finalize();return items

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='samples/smoke');p.add_argument('--frames',type=int,default=4)
    p.add_argument('--width',type=int,default=320);p.add_argument('--height',type=int,default=192)
    a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    items=sum((render(root,kind,a.frames,a.width,a.height) for kind in ['portrait','forest','village']),[])
    with (root/'manifest.jsonl').open('w') as f:
        for item in items:f.write(json.dumps(item)+'\n')
    print(f'Generated {len(items)} REAL-3D technical fixture frames: {root}')
if __name__=='__main__':main()
