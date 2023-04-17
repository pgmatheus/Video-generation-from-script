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
#
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

#
import shutil

#
import re

#example usage:

# high img res
""" from scripts_2.utilities import high_img
high_img_f = high_img(4)

high_img_f.run(
face = False, # reconstruct face
path = f'F:/gg/templates/to_proccess/rave party, beautiful --neg (blur)/0000000/interpolated_frames_film_a/{file}', # image path
w_i = 512, # width to be resized 
h_i = 512, # height to be resized
w_o = 1920, # width to be resized 
h_o = 1080, # height to be resized    
out_path = f'./test/{file}', # image output location
return_out = 'nothing', # return the data to use later
show_image = False, # show output
) """

# interpolated frames

""" from scripts_2.utilities import interp_fram

interp_fram.run(
       times_to_interpolate= 2, #multiplier of the frames
       path ='F:/gg/templates/to_proccess/rave party, beautiful --neg (blur)/0000000', #input folder with pngs
       out_path= 'F:/gg/test_inter', # output folder
       w_o= 512, # width output
       h_o= 512, # height output 
) """

# video from frames

""" from scripts_2.utilities import create_video_from_pngs
create_video_from_pngs("./test",f"./result3.mp4",30) """

# merge video with audio
""" from scripts_2.utilities import merge_video_with_audio

merge_video_with_audio('F:/gg/result3.mp4','F:/gg/musics/M000001.wav','F:/gg/merged_video2.mp4')
"""

# copy_file_and_rename

""" 
from scripts_2.utilities import copy_file_and_rename

copy_file_and_rename(path_first_img,temp_folder,'a_000000001.png')
file_input location, file_output location and new name"""

# def retrieve_frames(path_input_video, path_output_frames)

""" from scripts_2.utilities import retrieve_frames

retrieve_frames(path_input_video, path_output_frames) """

# def remove_temp_folder(temp_folder):

""" from scripts_2.utilities import remove_temp_folder

remove_temp_folder(temp_folder) """

# from scripts_2.utilities import split_text_to_dict

""" output = split_text_to_dict(txt) """


""" def create_video_from_pngs(directory_path, output_file_path, fps=60, w = 1920, h = 1080):
    image_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    if not image_paths:
        print('No PNG images found in directory:', directory_path)
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (w, h), True)
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image,(w,h))
        video_writer.write(image)
    video_writer.release() """

def create_video_from_pngs(directory_path, output_file_path, music_path='', temp_video= 'F:/gg/templates/temp_video.mp4', fps=60, w = 1920, h = 1080):
    image_paths = []
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    if not image_paths:
        print('No PNG images found in directory:', directory_path)
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if music_path == '':
        video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (w, h), True)
    else:
        if os.path.exists(temp_video):
            os.remove(temp_video)
        video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h), True)
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image,(w,h))
        video_writer.write(image)
    video_writer.release()

    if music_path != '':
        video_clip = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(music_path)

        # set the duration of the final video to be the same as the video clip
        duration_audio = audio_clip.duration
        duration_video = video_clip.duration

        if duration_audio <= duration_video:
            final_duration = duration_audio
        else:
            final_duration = duration_video

        #final_duration = math.floor(final_duration)

        # Define the FFmpeg command with -shortest option
        command = f"ffmpeg -i {temp_video} -i {music_path} -c:v copy -c:a aac -strict experimental -t {final_duration} {output_file_path}"

        # Execute the FFmpeg command using subprocess
        subprocess.call(command, shell=True)




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
        

    def run(self, face, path, w_i, h_i, w_o, h_o, out_path, return_out, show_image):
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (w_i,h_i))
        if face:                    
            _, _, output = self.face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(frame, outscale=self.upscale)
        if w_i*self.upscale !=  w_o or h_i*self.upscale != h_o:     
            output = cv2.resize(output,(w_o,h_o))
        cv2.imwrite(out_path, output)
        if show_image:
            plt.imshow(output)
            plt.show()
        if return_out == '':
            return output

    def __del__(self):
        torch.cuda.empty_cache()
        gc.collect()
        del self.upsampler
        del self.model
        del self.face_enhancer
        del self.upscale
        print('delete')



def load_images_from_folder(folder_path, w_o, h_o):
    images = []
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_url = os.path.join(folder_path, filename)
            image_data = tf.io.read_file(img_url)
            image = tf.io.decode_image(image_data, channels=3)
            image_numpy = tf.cast(image, dtype=tf.float32).numpy() #.numpy()
            """ image_numpy = tf.image.resize(
                images=image_numpy,
                size=[ h_o,  w_o],
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False,
                antialias=True,
            ).numpy() """
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
  
def __del__(self):

    del self._model
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()

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
    
    def run(path, w_o, h_o, out_path, times_to_interpolate):
        if not times_to_interpolate:
           times_to_interpolate = 1


        interpolator = Interpolator()
        #print(torch.cuda.memory_summary())
        input_frames = load_images_from_folder(path, w_o, h_o)
        #print(torch.cuda.memory_summary())
        print('interpolating frames')
        frames = list(
                interpolate_recursively(input_frames, times_to_interpolate,
                interpolator))
        print(f'video with {len(frames)} frames')
        """ print('saving frames')
        for i, single_frame in tqdm(enumerate(frames)):
            media.write_image(f"{out_path}/output_{i:04d}.png", single_frame) """
        
        media.show_video(frames, fps=60, title='FILM interpolated video')
        media.write_video(f"{out_path}/output.mp4", frames, fps=60)
        del interpolator
        frames = []
        input_frames = []
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()

def merge_video_with_audio(video_path, audio_path, output_path):
    # create VideoFileClip and AudioFileClip objects
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # set the duration of the final video to be the same as the video clip
    duration_audio = audio_clip.duration
    duration_video = video_clip.duration

    if duration_audio <= duration_video:
       final_duration = duration_audio
    else:
       final_duration = duration_video

    # set the audio of the video clip to the audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # create a CompositeVideoClip object with the video clip
    final_clip = CompositeVideoClip([video_clip.set_duration(final_duration)])

    # write the final clip to a file
    final_clip.write_videofile(output_path, threads=4, preset='slow')



def copy_file_and_rename(file_path,destination_folder,new_name):
    shutil.copy(file_path, destination_folder +'/'+ new_name)



def retrieve_frames(path_input_video, path_output_frames): 
    # Open the video file
    video = cv2.VideoCapture(path_input_video)

    # Initialize frame counter
    frame_count = 0

    # Loop through each frame in the video
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Save the frame as an image file
        cv2.imwrite(f"{path_output_frames}/a_{frame_count:07d}.png", frame)


        # Increment the frame counter
        frame_count += 1

    # Release the video file
    video.release()

def remove_temp_folder(temp_folder):
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)


def split_text_to_dict(text):
    lines = filter(None, re.split(r'(?<=[.!?])\s', text))  # Split text using regular expression to match dots, exclamation marks, and question marks followed by whitespace, and filter out empty lines
    result = {}
    for i, line in enumerate(lines):
        result[str(i)] = line.strip()  # Add line to the dictionary with the key as the line number and the value as the line text
    result.pop(str(len(result)-1), None)  # Remove the last item from the dictionary
    return result