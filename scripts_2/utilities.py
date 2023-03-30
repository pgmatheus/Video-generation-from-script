import torch
import gc
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import matplotlib.pyplot as plt
from tqdm import tqdm
#
import tensorflow as tf
import tensorflow_hub as hub # pip install --upgrade tensorflow-hub
import soundfile as sf
import requests
import numpy as np
from typing import Generator, Iterable, List, Optional
import mediapy as media #3 pip install mediapy
import os
import random
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, ImageSequenceClip, CompositeAudioClip
import subprocess
import ffmpeg
import cv2
from pydub import AudioSegment
import math
import googletrans
from googletrans import Translator

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
            output = cv2.resize(output,(obj['w_o'],obj['h_o']))
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


def load_images_from_folder(folder_path, w_o, h_o):
    images = []
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_url = os.path.join(folder_path, filename)
            image_data = tf.io.read_file(img_url)
            image = tf.io.decode_image(image_data, channels=3)
            image_numpy = tf.cast(image, dtype=tf.float32) #.numpy()
            image_numpy = tf.image.resize(
                images=image_numpy,
                size=[ h_o,  w_o],
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False,
                antialias=True,
            ).numpy()
            images.append(image_numpy / _UINT8_MAX_F)
    return images

def _pad_to_align(x, align):

  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop
class Interpolator:

  def __init__(self, align: int = 64) -> None:

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    self._model = hub.load(".\\film_1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:

    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()

def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator) -> Generator[np.ndarray, None, None]:

  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)

def interpolate_recursively(
    frames: List[np.ndarray], num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:

  n = len(frames)
  
  for i in tqdm(range(1, n)):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    num_recursions, interpolator)
  # Separately yield the final frame.
  yield frames[-1]


class interp_fram:
    def run(obj):
        times_to_interpolate = obj['times_to_interpolate'] or 1

        interpolator = Interpolator()
        
        input_frames = load_images_from_folder(obj['path'], obj['w_o'], obj['h_o'])
        print('interpolating frames')
        frames = list(
                interpolate_recursively(input_frames, times_to_interpolate,
                interpolator))
        print(f'video with {len(frames)} frames')
        print('saving frames')
        for i, single_frame in tqdm(enumerate(frames)):
            media.write_image(f"{obj['out_path']}/output_{i:04d}.png", single_frame)

        media.show_video(frames, fps=60, title='FILM interpolated video')
        del interpolator
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gc.collect()