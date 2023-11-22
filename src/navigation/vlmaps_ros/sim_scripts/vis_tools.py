# @title Helper functions for video creation and display
import sys
import os
import imageio
import numpy as np
import cv2
import tqdm
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth

def get_fast_video_writer(video_file: str, fps: int = 60):

    if "google.colab" in sys.modules:
        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

    # @markdown if the colab instance doesn't have GPU, untick the following checkbox
    has_gpu = True # @param {type: "boolean"}
    codec = "h264"
    if has_gpu:
        codec = "h264_nvenc"


    if (
        "google.colab" in sys.modules
        and os.path.splitext(video_file)[-1] == ".mp4"
        and os.environ.get("IMAGEIO_FFMPEG_EXE") == "/usr/bin/ffmpeg"
    ):
        # USE GPU Accelerated Hardware Encoding
        writer = imageio.get_writer(
            video_file,
            fps=fps,
            codec=codec,
            mode="I",
            bitrate="1000k",
            format="FFMPEG",
            ffmpeg_log_level="info",
            quality=10,
            output_params=["-minrate", "500k", "-maxrate", "5000k"],
        )
    else:
        # Use software encoding
        writer = imageio.get_writer(video_file, fps=fps)
    return writer

def create_video(data_dir: str, output_dir: str, fps: int = 30):
    
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    rgb_out_path = os.path.join(output_dir, "rgb.mp4")
    depth_out_path = os.path.join(output_dir, "depth.mp4")
    rgb_writer = get_fast_video_writer(rgb_out_path, fps=fps)
    depth_writer = get_fast_video_writer(depth_out_path, fps=fps)

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
        x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pbar = tqdm.tqdm(total=len(rgb_list), position=0, leave=True)
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = load_depth(depth_path)
        depth_vis = (depth / 10 * 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        rgb_writer.append_data(rgb)
        depth_writer.append_data(depth_color)
        pbar.update(1)
    rgb_writer.close()
    depth_writer.close()

    #print save path
    print("RGB video saved at: ", rgb_out_path)
    print("Depth video saved at: ", depth_out_path)

def show_video(video_path, video_width = 1080):
   
    video_file = open(video_path, "r+b").read()
  
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={video_width} autoplay controls><source src="{video_url}"></video>""")

def show_videos(video_paths, video_width = 1080):
    html = ""
    for video_path in video_paths:
        video_file = open(video_path, "r+b").read()
      
        video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
        html += f"""<video width={video_width} autoplay controls><source src="{video_url}"></video>
                 """
    return HTML(html)

def show_obstacle_map(obstacles:np.ndarray):

    x_indices, y_indices = np.where(obstacles == 0)
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)
    print(np.unique(obstacles))

    obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap='gray')
    plt.title("Obstacle Map")
    plt.show()

def show_color_top_down(color_top_down:np.ndarray,obstacles:np.ndarray):


    x_indices, y_indices = np.where(obstacles == 0)
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)
    print(" No of unique obstacles are", np.unique(obstacles))
    color_top_down = color_top_down[xmin:xmax+1, ymin:ymax+1]
        
    color_top_down_pil = Image.fromarray(color_top_down)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.title("Color Top Down View")
    plt.show()

def show_seg_mask(seg:np.ndarray, patches:list):

    seg = Image.fromarray(seg)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps Inferences")
    plt.imshow(seg)
    plt.show()

def show_bounding_boxes(outputs:dict,labels:list):

    for label in labels:
        mask = outputs[label]['mask'].astype(np.uint8)
        assert mask.dtype == np.uint8
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        
        for bbox in outputs[label]['bboxes']:
            center = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
            cv2.rectangle(mask,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            cv2.circle(mask,center,5,(0,255,0),-1)
            
        plt.figure()
        plt.imshow(mask)
        plt.title(label)
        plt.show()

def show_label_masks(mask_list:list,labels:list):

    for i in range(len(labels)):
        plt.figure()
        plt.imshow(mask_list[i],cmap='gray')
        plt.title('Mask for '+labels[i])
        plt.show()
