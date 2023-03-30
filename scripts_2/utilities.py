import torch
import gc
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import matplotlib.pyplot as plt


#example usage:

# high img res
""" from scripts_2.utilities import high_img
high_img_f = high_img(4)

high_img_f.run({
    'face': False, # reconstruct face
    'path': f'./test_inp/{file}', # image path
    'w_i': 480, # width to be resized 
    'h_i': 270, # height to be resized
    'w_o': 1920, # width output 
    'h_o': 1080, # height output    
    'out_path': f'./test/{file}', # image output location
    'return_out': 'nothing', # return the data to use later
    'show_image': False, # show output
}) """

class high_img:
    def __init__(self, upscale):        
        netscale = 4
        model_path_face = "./realesrgan/GFPGANv1.3.pth"
        dni_weight = None
        tile = 0
        title_pad = 10
        pre_pad = 0
        half = None
        gpu_id = None
        self.upscale = upscale
        img_mode = 'RGBA'
        # R-ESRGAN + Anime
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path_x4 = "./realesrgan/RealESRGAN_x4plus_anime_6B.pth"
        self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path_x4,
                dni_weight=dni_weight,
                model=self.model,
                tile=tile,
                tile_pad=title_pad,
                pre_pad=pre_pad,
                half= not half,
                gpu_id= gpu_id
        )

        self.face_enhancer = GFPGANer(
                    model_path=model_path_face,
                    upscale=self.upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler)
        

    def run(self, obj):
        frame = cv2.imread(obj['path'], cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (obj['w_i'],obj['h_i']))
        if obj['face']:                    
            _, _, output = self.face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(frame, outscale=self.upscale)
        if obj['w_i']*self.upscale !=  obj['w_o'] or obj['h_i']*self.upscale != obj['h_o']:     
            output = cv2.resize((obj['w_o'],obj['h_o']))
        cv2.imwrite(obj['out_path'], output)
        if obj['show_image']:
            plt.imshow(output)
            plt.show()
        if obj['return_out'] == '':
            return output

    def __del__(self):
        del self.upsampler
        del self.model
        del self.face_enhancer
        del self.upscale
        print('delete')
        torch.cuda.empty_cache()
        gc.collect()


class interp_fram:
    def __init__(self):