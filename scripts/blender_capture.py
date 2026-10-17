"""Capture a supplied .blend scene as multilayer EXR, using Blender's own Python.

Example:
  blender -b /path/to/existing_scene.blend --python scripts/blender_capture.py -- \
    --out /data/shot001 --start 1 --end 24 --width 1280 --height 720

Unexecuted here because Blender is not installed in the CPU sandbox.
"""
import argparse,sys
from pathlib import Path
import bpy

def arguments():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--samples',type=int,default=16)
    return p.parse_args(argv)

def main():
    a=arguments()
    if a.end<a.start:raise ValueError('--end before --start')
    s=bpy.context.scene
    if not s.camera:raise RuntimeError('Scene has no active camera')
    s.render.engine='CYCLES';s.cycles.samples=a.samples
    s.render.resolution_x=a.width;s.render.resolution_y=a.height;s.render.resolution_percentage=100
    s.render.use_motion_blur=False
    s.render.image_settings.file_format='OPEN_EXR_MULTILAYER'
    s.render.image_settings.color_mode='RGBA';s.render.image_settings.color_depth='32'
    s.render.image_settings.exr_codec='ZIP'
    view=bpy.context.view_layer
    view.use_pass_z=True;view.use_pass_normal=True;view.use_pass_vector=True
    view.use_pass_cryptomatte_object=True;view.use_pass_cryptomatte_material=True
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for f in range(a.start,a.end+1):
        s.frame_set(f)
        s.render.filepath=str(out/f'frame_{f:06d}.exr')
        bpy.ops.render.render(write_still=True)
        print('CAPTURED:',s.render.filepath)
if __name__=='__main__':main()
