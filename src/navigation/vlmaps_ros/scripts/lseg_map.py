import argparse
import math
import os
import time
import clip
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import lseg
from lseg.additional_utils.models import crop_image, pad_image, resize_image
from lseg.modules.models.lseg_net import LSegEncNet
from utils.clip_mapping_utils import *
from utils.clip_mapping_utils import get_tf_camera_optical2pcd
from utils.time_utils import Tic


def create_lseg_map_batch(img_save_dir, use_self_built_map: False ,camera_height,max_depth_value=5, cs=0.05, gs=1000, depth_sample_rate=100,device = "cuda"):
    """ Creates a map using the LSeg model."""
    
    mask_version = 1  # 0, 1
    crop_size = 480  # 480
    # base_size = 520  # 520
    base_size = 848  # 520
    lang = "door,chair,ground,ceiling,other"
    labels = lang.split(",")
    vis = False

    # loading models
    device = device if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
    }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels)
    lang_token = lang_token.to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    model = LSegEncNet(lang, arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    model_state_dict = model.state_dict()
    checkpoint_file = os.path.join(img_save_dir,"../..","scripts/lseg/checkpoints/demo_e200.ckpt")
    pretrained_state_dict = torch.load(checkpoint_file)
    pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.cuda()

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    print(f"loading scene {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    pose_dir = os.path.join(img_save_dir, "pose")
    semantic_dir = os.path.join(img_save_dir, "semantic")
    obj2cls_path = os.path.join(img_save_dir, "obj2cls_dict.txt")
    config_dir = os.path.join(img_save_dir, "../..", "config/data_configs/")
    add_params_dir = os.path.join(config_dir, "add_params")

    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    semantic_list = sorted(os.listdir(semantic_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]
    semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]

    # Get camera intrinsics path, base2cam tf path
    tail = os.path.split(img_save_dir)[-1]
    tf_base2cam_path = os.path.join(add_params_dir, f"{tail}_0tf_base2cam.txt")
    cam_intrinsics_path = os.path.join(add_params_dir, f"{tail}_0cam_intrinsics.txt") 

    map_save_dir = os.path.join(img_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    obj2cls = load_obj2cls_dict(obj2cls_path)

    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    gt = np.zeros((gs, gs), dtype=np.int32)
    grid = np.zeros((gs, gs, clip_feat_dim), dtype=np.float32)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    weight = np.zeros((gs, gs), dtype=float)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

    tf_list = []
    data_iter = zip(rgb_list, depth_list, semantic_list, pose_list)
    pbar = tqdm(total=len(rgb_list))
    # load all images and depths and poses
    for idx,data_sample in enumerate(data_iter):
        rgb_path, depth_path, semantic_path, pose_path = data_sample
        st = time.time()

        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        assert crop_size == rgb.shape[1] and base_size == rgb.shape[0], "Pixel feature size not equal to rgb!! RGB shape = {}, Pix shape = {}".format(
            rgb.shape, (crop_size, base_size)
        )

        # read pose
        pos, rot = load_pose(pose_path)  # x forward, y left, z up (Stretch default frame)
        tf_base2cam = load_tf(tf_base2cam_path)
        tf_camera_optical2pcd = get_tf_camera_optical2pcd()
        intrinsics = load_intrinsics(cam_intrinsics_path)

        # Convert intrinsics to account for stretch's 90 deg rotated camera
        intrinsics = convert_intrinsics(intrinsics, rgb.shape[0], rgb.shape[1])

        base_pose = np.eye(4)
        base_pose[:3, :3] = rot
        base_pose[:3, 3] = pos.reshape(-1)

        # camera pose w.r.t base frame (z forward, x right, y down)
        pose = base_pose @ tf_base2cam @ tf_camera_optical2pcd

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])

        # The initial pose frame is w.r.t to some world coordinate frame. RTABMAP output.
        # This operation converts poses from world frame to the initial frame!
        tf = init_tf_inv @ pose

        # read depth
        depth = load_depth(depth_path)

        # WARNING: Do not resize depth image since it changes intrinsic matrix and interpolation gives bad depths
        # depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth = depth.astype(np.float32) / 1000.0 # mm to m
        assert depth.shape == rgb.shape[:2], "depth shape is not equal to rgb shape"

        # read semantic
        semantic = load_semantic(semantic_path)
        semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        # What is the relevance of basesize/crop size w.r.t quality of lseg features?
        pix_feats = get_lseg_feat(model, rgb.copy(), labels, transform, crop_size, base_size, norm_mean, norm_std, vis, device=device)

        # project all point cloud onto the ground
        if(use_self_built_map):
            pc, mask = depth2pc_real(depth,max_depth_value=max_depth_value,intrinsics=intrinsics)

        else:
            pc, mask = depth2pc(depth,max_depth_value=max_depth_value)

        # shuffle the point cloud
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        # sample 'depth_sample_rate' points randomly from shuffle_mask
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask] # from all the sampled points. Select only those that are farther than 0.1m
        
        # Transform the point cloud to the initial frame
        # Remember point cloud is always computed in z forward, y downward, x to the right frame
        # The rot matrix we computed earlier takes it into consideration

        # The point cloud is computed in z forward, y downward, x to the right frame
        pc_global = transform_pc(pc, tf) # transform to the initial frame

        # maximum values of x,y,z in the global frame
        print("range x,y,z in global frame: ", np.max(pc_global, axis=1)-np.min(pc_global, axis=1))


        if(use_self_built_map):
            if(intrinsics is not None):
                rgb_cam_mat = intrinsics
                feat_cam_mat = intrinsics
            else:
                rgb_cam_mat = get_real_cam_mat(rgb.shape[0], rgb.shape[1])
                feat_cam_mat = get_real_cam_mat(pix_feats.shape[2], pix_feats.shape[3])
        else:
            rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
            feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            #p_local is in z forward, y downward, x to the right frame
            # but the image itself is rotated. So we need to rotate the point cloud as well
            if i % 1 != 0:
                continue
            x, y = pos2grid_id(gs, cs, p[0], p[2])
            # print("Grid points = ",(x,y))
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or x < 0 or y < 0 or p_local[1] < -0.5: # removing ceiling points maybe @atharva
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            semantic_v = semantic[rgb_py, rgb_px]
            if semantic_v == 40:
                semantic_v = -1

            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
                gt[y, x] = semantic_v

            px, py, pz = project_point(feat_cam_mat, p_local) # project local pc back to the image
            # discard points outside the image
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]): 
                feat = pix_feats[0, :, py, px] # typical (B,C,H,W) format
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            if p_local[1] > camera_height: # Removing floor points maybe @atharva
                continue
            obstacles[y, x] = 0

        et = time.time()
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)


def get_lseg_feat(
    model: LSegEncNet,
    image: np.array,
    labels,
    transform,
    crop_size=480,
    base_size=520,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.5, 0.5, 0.5],
    vis=False,
    device="cuda:1" if torch.cuda.is_available() else "cpu",
):
    print("image shape: ", image.shape)
    vis_image = image.copy()
    image = transform(image).unsqueeze(0).to(device)
    img = image[0].permute(1, 2, 0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width, **{"mode": "bilinear", "align_corners": True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape  # .size()
        assert ph >= height and pw >= width
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().to(device)
                logits_outputs = image.new().resize_(batch, len(labels), ph, pw).zero_().to(device)
            count_norm = image.new().resize_(batch, 1, ph, pw).zero_().to(device)
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                count_norm[:, :, h0:h1, w0:w1] += 1
        assert (count_norm == 0).sum() == 0
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:, :, :height, :width]
        logits_outputs = logits_outputs[:, :, :height, :width]
    outputs = outputs.cpu()
    outputs = outputs.numpy()  # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]
    if vis:
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")
        cv2.imshow("image", vis_image[:, :, [2, 1, 0]])
        cv2.waitKey()
        fig = plt.figure()
        plt.imshow(seg)
        plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 20})
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The directory of the rgb, depth, and pose")
    parser.add_argument("--camera_height", type=float, default=1.5, help="The height of the camera in the simulator")
    parser.add_argument("--depth_sample_rate", type=int, default=100, help="The height of the camera in the simulator")
    args = parser.parse_args()
    create_lseg_map_batch(args.data_dir, args.camera_height, depth_sample_rate=args.depth_sample_rate)
