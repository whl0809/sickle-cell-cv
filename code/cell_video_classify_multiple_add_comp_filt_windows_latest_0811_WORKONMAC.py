print("DEBUG: IMPORT - 1")
import cv2
import torch
import torch.nn as nn
import numpy as np
print("DEBUG: IMPORT - 2")
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTModel,AutoModel
print("DEBUG: IMPORT - 3")
from PIL import Image
import os
from cellpose import models, utils
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
print("DEBUG: IMPORT - 4")
from collections import defaultdict,Counter,deque
from cellpose import plot
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops
import argparse
import os
import pickle

#------------- Constants ----------------
# Note (kaiyu): in cv2, color is BGR
BLUE = (255, 0, 0)
RED = (0, 0, 255)
LABEL_CHANGED = 0
LABEL_UNCHANGED = 1

# Note (kaiyu): Debug print
DEBUG_MODE = False
def DEBUG_PRINT(msg, *args):
    if DEBUG_MODE:
        print(f"DEBUG: {msg}", *args)

print("-------------------- Parameterization --------------------")
parser = argparse.ArgumentParser(description='Cell video classifier with filtering and visualization.')

parser.add_argument('-i', '--inputs', type=str, required=True,
                    help='Comma-separated list of input video files, e.g., v1.mov,v2.mov,v3.mov')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Output directory')
parser.add_argument('--frame_skip', type=int, default=2,
                    help='Process every Nth frame')
parser.add_argument('--max_frame', type=int, default=480,
                    help='Max number of frames to process')

args = parser.parse_args()

video_paths = args.inputs.split(',')
all_out = args.output_dir
frame_skip = args.frame_skip
max_frame = args.max_frame
output = [os.path.splitext(os.path.basename(v))[0] + '.avi' for v in video_paths]
out_path = [os.path.join(all_out, os.path.splitext(os.path.basename(v))[0]) for v in video_paths]
fps = 4 # 4 frames - 1 second
#print(video_paths,all_out,output,out_path)
#exit()
os.makedirs(all_out, exist_ok=True)
for op in out_path:
    os.makedirs(op, exist_ok=True)

# Removes edge cells that fall below a specified threshold
def remove_edge_cells(masks, threshold=0.3):
    unique_ids = range(1, np.max(masks)+1)

    # average cell takes up ~1500 slots in an (729, 1094) ndarray
    avg_cell_area = sum([np.sum(masks == cid) for cid in unique_ids]) / len(unique_ids)

    # generates list of edges to be removed
    top    = masks[0, :]
    bottom = masks[-1, :]
    left   = masks[:, 0]
    right  = masks[:, -1]
    border_pixels = np.concatenate([top, bottom, left, right])
    edge_ids = np.unique(border_pixels[border_pixels > 0])
    remove_edge_list = []
    for edge_id in edge_ids:
        if np.sum(masks == edge_id) < threshold*avg_cell_area:
            remove_edge_list.append(edge_id)

    # generates new set of masks without targeted edge cells
    filtered_masks = np.zeros(masks.shape)
    cellidnumber = 1
    for cid in unique_ids:
        if cid not in remove_edge_list:
            filtered_masks += ((masks == cid)*cellidnumber)
            cellidnumber += 1
    
    return filtered_masks
    

dname={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F'}
    
def plot_total_binary_ratio(df, out_path, frame_skip, fps ,title='Total cell ration (binary)'):
    total_pos = df[[f'Class_{i}_pos' for i in range(6)]].sum(axis=1)
    total_count = df[[f'Class_{i}_total' for i in range(6)]].sum(axis=1)
    total_ratio = 1 - total_pos / total_count.replace(0, np.nan)
    time_sec = df['FrameIndex'] * frame_skip / fps
    plt.figure(figsize=(8, 5))
    y_percent = total_ratio * 100
    plt.plot(time_sec, y_percent, label='Total sickled fraction', color='black')
    #plt.xlabel('Frame Index')
    plt.xlabel('Time (s)')
    plt.ylabel('Sickled fraction (%)')
    plt.ylim(0,100)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    # === 新增：保存每个点到 CSV ===
    csv_out_path = out_path.replace(".png", ".csv")  # 输出文件名与图同名
    df_out = pd.DataFrame({
        "Time_sec": time_sec,
        "Sickled_fraction_percent (%)": y_percent
    })
    df_out.to_csv(csv_out_path, index=False)
    print(f"Saved curve data to {csv_out_path}")
    

# ===  For cell tracing - Optical Flow function ===
def estimate_next_bboxes(prev_frame_gray, curr_frame_gray, prev_bboxes):
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray,
                                        None, 0.5, 3, 21, 3, 5, 1.2, 0)
    updated_bboxes = {}
    for cid, bbox in prev_bboxes.items():
        x, y, w, h = bbox
        region_flow = flow[y:y+h, x:x+w]
        if region_flow.size == 0:
            updated_bboxes[cid] = bbox
            continue
        dx = np.mean(region_flow[..., 0])
        dy = np.mean(region_flow[..., 1])
        nx, ny = int(x + dx), int(y + dy)
        updated_bboxes[cid] = (nx, ny, w, h)
    return updated_bboxes

def resize_frame(frame, ratio):
    h, w = frame.shape[:2]
    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def upscale_bbox(bbox, ratio):
    x, y, w, h = bbox
    return tuple(int(coord / ratio) for coord in (x, y, w, h))


def segment_frame_downscaled_ds(original_frame, model_path, out_path, ratio=0.2, diameter=30,is_frame_0=False):
    """
    修正后的下采样分割函数，确保坐标正确映射回原图
    """
    # 记录原图尺寸
    orig_h, orig_w = original_frame.shape[:2]

    # 计算新尺寸并缩放
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 加载模型并运行Cellpose
    cellpose_model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    masks, flows, styles = cellpose_model.eval(resized_frame, diameter=diameter, channels=[0, 0])

    # Debug only for frame 0
    if is_frame_0 :

        plt.imshow(masks, cmap='gray')
        plt.title('Cellpose masks (BEFORE remove edge cells)')
        plt.axis('off')
        plt.imsave(out_path+"/masks_BEFORE_remove_edge_cells.png", masks, cmap="gray")

    # For debug
    #print(frame_idx)
    # if frame_idx>95:
    #     fig = plt.figure(figsize=(8, 4))
    #     plot.show_segmentation(fig, resized_frame, masks, flows[0], channels=[0, 0])
    #     plt.title("Crop Cellpose Segmentation (before_edge_cells)")
    #     plt.show()

    
    masks = remove_edge_cells(masks)

    # Debug only for frame 0    
    if is_frame_0:
        DEBUG_PRINT("Saving masks.npy before remove_edge_cells")
        savepath = out_path+"/masks_remove_edge_cells.npy"
        np.save(savepath, masks)
        plt.imshow(masks, cmap='gray')
        plt.title('Cellpose masks (AFTER remove edge cells)')
        plt.axis('off')
        plt.imsave(out_path+"/masks_AFTER_remove_edge_cells.png", masks, cmap="gray")
        DEBUG_PRINT("masks.npy AFTER remove_edge_cells saved at", savepath)
        DEBUG_PRINT("    number of zero elements;", np.count_nonzero(masks == 0))

        #fig = plt.figure(figsize=(8, 4))
        #plot.show_segmentation(fig, resized_frame, masks, flows[0], channels=[0, 0])
        #plt.title("Crop Cellpose Segmentation (remove_edge_cells)")
        #plt.show()

    # 提取边界框并映射回原图坐标
    unique_ids = np.unique(masks)[1:]  # 跳过背景
    bboxes = {}
    for cid in unique_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # 获取缩放后的边界框
        x, y, w, h = cv2.boundingRect(contours[0])

        # 映射回原图坐标
        x_orig = int(x / ratio)
        y_orig = int(y / ratio)
        w_orig = int(w / ratio)
        h_orig = int(h / ratio)

        bboxes[cid] = (x_orig, y_orig, w_orig, h_orig)

    return bboxes, masks, unique_ids, resized_frame

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def check_size_outline(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    #tolerance = 1e-6
    fold=5
    c1=0
    c2=0
    if w2==0 or h2==0:
        return True
    if w1>w2:
        if w1-fold*w2>0:
            c1=1
    else:
        if w2-fold*w1>0:
            c1=1
    if h1>h2:
        if h1-fold*h2>0:
            c2=1
    else:
        if h2-fold*h1>0:
            c2=1
    #print(w1,w2)
    #print(h1,h2)
    #print(c1,c2)
    if c1==1 and c2==1:
        return True
    else:
        return False
    '''
    if (abs(w1 / w2 - fold) < tolerance or abs(w2 / w1 - fold) < tolerance) and (abs(h1 / h2 - fold) < tolerance or abs(h2 / h1 - fold) < tolerance):
        return True
    else:
        return False
    '''

def check_pos_outline_iter(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1-x2)>=200 and abs(y1-y2)>=200:
        return True
    # elif  abs(x1-x2)>=200  or abs(y1 - y2) >= 200:
    #     return True
    # elif abs(x1-x2)>=200 and abs(y1-y2)>=50:
    #     return True
    # elif abs(y1 - y2) >= 200 and abs(x1 - x2) >= 50:
    #     return True
    else:
        return False


def check_pos_outline(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1-x2)>=100 or abs(y1-y2)>=100:
        return True
    # elif  abs(x1-x2)>=200  or abs(y1 - y2) >= 200:
    #     return True
    # elif abs(x1-x2)>=200 and abs(y1-y2)>=50:
    #     return True
    # elif abs(y1 - y2) >= 200 and abs(x1 - x2) >= 50:
    #     return True
    else:
        return False


# -------------------- 中心点距离 --------------------
def center_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    #cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    #cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    cx1, cy1 = x1, y1
    cx2, cy2 = x2, y2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


# -------------------- 追踪函数（中心点+IoU） --------------------
def match_cells_tracking(prev_cells, curr_masks, bboxes):
    matches = {}
    unmatched = list(np.unique(curr_masks))
    if 0 in unmatched:
        unmatched.remove(0)  # remove background
    used_curr = set()

    for track_id, prev_info in prev_cells.items():
        prev_frame_index = prev_info['latest_frame_index']
        prev_box = prev_info['bbox'][prev_frame_index]
        min_dist = float('inf')
        best_iou = 0
        best_cid = None
        best_box = None
        #print(track_id,prev_box)
        #exit()

        for cid in unmatched:
            x, y, w, h = bboxes[cid][0], bboxes[cid][1], bboxes[cid][2], bboxes[cid][3]
            curr_box = (x, y, w, h)
            dist = center_distance(prev_box, curr_box)
            iou = compute_iou(prev_box, curr_box)
            # if check_pos_outline_iter(prev_box, curr_box):continue

            #if not prev_cells[int(cid)]['class']==prev_cells[int(track_id)]['class']:continue
            if dist < min_dist:  # 结合距离与IoU
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box
            elif dist < 50 and iou > best_iou:
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box


        # check_id=105
        # if int(track_id) == check_id:
        #     print(best_box)

        # Note (kaiyu): other fields are removed since not used
        if best_cid is not None:
            if check_size_outline(prev_box, best_box) or check_pos_outline(prev_box, best_box):
                matches[track_id] = {
                    'bbox': prev_box,
                    'class': prev_info['class'],
                }
                # if int(track_id) == check_id:
                #     #print(cid, 'doesn\'t pass check_outline')
                #     print('best_cid is None - ', prev_box)
            else:

                matches[track_id] = {
                    'bbox': best_box,
                    'class': prev_info['class'],
                }
                # if int(track_id)==check_id:
                #     print('best_cid is not None - ',best_box)
                # used_curr.add(best_cid)
        else:
            matches[track_id]={
                'bbox': prev_box,
                'class': prev_info['class'],
            }
            # if int(track_id)==check_id:
            #     print('best_cid is None - ', prev_box)

    return matches, used_curr

# -------------- Aspect Ratio Calculation -----------
def aspect_ratio(mask):
    # intakes binary mask of cell, outputs aspect ratio
    if mask.sum() == 0:
        return 0.0  # Empty mask

    labeled = label(mask)
    props = regionprops(labeled)

    if len(props) == 0:
        return 0.0

    region = props[0]
    major = region.major_axis_length
    minor = region.minor_axis_length

    if minor == 0:
        return float('inf')  # Very thin shape
    return major / minor    


#------------------ Remove false positives ---------------
def remove_bin_label_false_positives(cell_info):
    for cid in cell_info:
        frame_indices = list(sorted(cell_info[cid]["state_history"].keys()))
        print(f"Removing false positives for {cid}")        
        # Imagine     B B B R R B B R R B R R R R
        # Make a box  B B B|R R B B R R B|R R R R
        #                   Left          Right
        # Count the number of R in the box (4)
        # Calculate R_count / (Right - Left)
        # Determine box_left
        box_left = 0   # list index to frame_indices
        for frame_index in frame_indices:
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_CHANGED:
                break
            box_left += 1

        # Determine box_right
        box_right = len(frame_indices) # len(cell_info[cid]["state_history"])  # list index to frame_indices
        for frame_index in reversed(frame_indices):
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_UNCHANGED:
                break
            box_right -= 1
            
        if box_right <= box_left:
            DEBUG_PRINT("[remove_false_positive] box_right <= box_left -- no false positive detected")
            continue

        # Count labels in the box
        changed_count = 0
        for list_index in range(box_left, box_right):
            frame_index = frame_indices[list_index]
            bin_label = cell_info[cid]["state_history"][frame_index]
            if bin_label == LABEL_CHANGED:
                changed_count += 1

        # Calculate ratio
        if changed_count / (box_right - box_left) > 0.5:
            for list_index in range(box_left, box_right):
                frame_index = frame_indices[list_index]
                cell_info[cid]["state_history"][frame_index] = LABEL_CHANGED
        else:
            for list_index in range(box_left, box_right):
                frame_index = frame_indices[list_index]
                cell_info[cid]["state_history"][frame_index] = LABEL_UNCHANGED            

# -------------------- Save intermediate results (in case we couldn't finish running for all videos) ---------------------
def save_intermediate_results(cell_info, df, out_path):
    """
    Saves cell_info, df after processing a video to out_path/cell_info.pkl and out_path/df.pkl
    
    cell_info: cell info collected after processing a video
    df: data frame after processing a video
    out_path: the output directory corresponding to an input video
    """
    os.makedirs(out_path, exist_ok=True)

    # Save cell_info dict
    cell_info_path = os.path.join(out_path, "cell_info.pkl")
    with open(cell_info_path, "wb") as f:
        pickle.dump(cell_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save pandas DataFrame
    df_path = os.path.join(out_path, "df.pkl")
    df.to_pickle(df_path, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved cell_info to {cell_info_path} and DataFrame to {df_path}")
                
    
# ========== 模型定义 ============
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.pooler_output
        return self.classifier(cls_token)
#exit()
# ========= 配置路径 =========
#video_path = 'test_video.mp4'
#output_video_path = 'output/annotated_video.avi'
six_class_model_path = 'best_model_vit_torch_macos.pth'
binary_model_path = 'best_model_vit_torch_macos_raw_vit_large_binary.pth'
binary_model_path_C = "direct_vit_C.pt"
binary_model_path_D = "direct_vit_E.pt"
binary_model_path_F = 'best_model_vit_torch_macos_raw_vit_large_binary_G.pth'
binary_model_path_F_brandon = "direct_vit_G.pt"

# ============ Device ==============
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Jianlu editted mps
#device='mps'
import torch

# Correct cross-platform device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
# ========= 加载模型 =========
six_class_model = ViTClassifier(num_classes=6)
six_class_model.load_state_dict(torch.load(six_class_model_path, map_location=device))
six_class_model.to(device)
six_class_model.eval()

binary_model = ViTClassifier(num_classes=2)
binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
binary_model.to(device)
binary_model.eval()

binary_model_C = ViTClassifier(num_classes=2)
binary_model_C.load_state_dict(torch.load(binary_model_path_C, map_location=device))
binary_model_C.to(device)
binary_model_C.eval()

binary_model_D = ViTClassifier(num_classes=2)
binary_model_D.load_state_dict(torch.load(binary_model_path_D, map_location=device))
binary_model_D.to(device)
binary_model_D.eval()

binary_model_F = ViTClassifier(num_classes=2)
binary_model_F.load_state_dict(torch.load(binary_model_path_F, map_location=device))
binary_model_F.to(device)
binary_model_F.eval()

binary_model_Fb = ViTClassifier(num_classes=2)
binary_model_Fb.load_state_dict(torch.load(binary_model_path_F_brandon, map_location=device))
binary_model_Fb.to(device)
binary_model_Fb.eval()


#six_class_model = torch.load(six_class_model_path, map_location=device).eval()
#binary_model = torch.load(binary_model_path, map_location=device).eval()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])
#exit()
def process_video(video_path, out_path, video_id, output_video_path,six_class_model,binary_model,feature_extractor,transform,cellpose_model_path = 'cyto3_train0327',frame_skip=frame_skip,max_frame=max_frame,fps=fps):
    # ========= 视频初始化 =========
    print('- Initialization......')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print('- Initialization......Done~')
    output_fps = fps / frame_skip

    # ========= 第0帧：分割 + 初始分类 =========
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Can not load the 0 time point from the video.")
    DEBUG_PRINT("Save first frame png")
    cv2.imwrite(out_path+"/first_frame.png", cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # Note (kaiyu):
    # We first run through the predictions, and only after all
    # predictions are made, we generate the video, so that we 
    # could remove false positives from the binary model's predictions.
    
    # ======= Process Cells =========
    DEBUG_PRINT("Run segment_frame_downscaled_ds w/ cellpose_model")
    bboxes, masks_seg, unique_ids, resized_frame = segment_frame_downscaled_ds( first_frame, cellpose_model_path, out_path,is_frame_0=True)

    # Note (kaiyu): cell_info is a SUPER IMPORTANT data structure; it holds all relevant
    # information output by the models to produce results
    cell_info = {}

    print('- Process cells in frame 0......')
    for cid in tqdm(unique_ids, desc='Process cells in frame 0 - progress:'):
        DEBUG_PRINT("Running Six Class Prediction for Cell ID: ", cid)

        mask = (masks_seg == cid).astype(np.uint8)
        x, y, w, h=bboxes[cid][0],bboxes[cid][1],bboxes[cid][2],bboxes[cid][3]
        cell_crop = first_frame[y:y+h, x:x+w]
        cell_pil = Image.fromarray(cell_crop)
        cell_tensor = transform(cell_pil).unsqueeze(0)

        with torch.no_grad():
            cls_output = six_class_model(cell_tensor.to(device))
            cls_probs = torch.softmax(cls_output, dim=1)
            cls_label = torch.argmax(cls_probs, dim=1).item()
            cls_prob = cls_probs[0, cls_label].item()

            # Apply aspect ratio threshold of 1.5 for binary classification of A/F
            if cls_label == 0 or cls_label == 5:
                if aspect_ratio(mask) >= 1.5:
                    cls_label = 5
                else:
                    cls_label = 0

            if cls_label==5:
                bin_output = binary_model_Fb(cell_tensor.to(device))
                bin_probs = 1-torch.softmax(bin_output, dim=1)
                bin_label = torch.argmax(bin_probs, dim=1).item()
                bin_prob = bin_probs[0, bin_label].item()
            elif cls_label==2:
                bin_output = binary_model_C(cell_tensor.to(device))
                bin_probs = 1 - torch.softmax(bin_output, dim=1)
                bin_label = torch.argmax(bin_probs, dim=1).item()
                bin_prob = bin_probs[0, bin_label].item()
            elif cls_label==3:
                bin_output = binary_model_D(cell_tensor.to(device))
                bin_probs = 1 - torch.softmax(bin_output, dim=1)
                bin_label = torch.argmax(bin_probs, dim=1).item()
                bin_prob = bin_probs[0, bin_label].item()
            else:
                bin_output = binary_model(cell_tensor.to(device))
                bin_probs = torch.softmax(bin_output, dim=1)
                bin_label = torch.argmax(bin_probs, dim=1).item()
                bin_prob = bin_probs[0, bin_label].item()

        # Note (kaiyu): saves all bbox and state detections, per frame where detection was made. 
        cell_info[cid] = {
            'bbox': {0: (x, y, w, h)},     # cell bounding box; maps from frame id to bbox
            'class': cls_label,            # predicted class based on frame0
            'class_prob': cls_prob,        # prob of predicted class
            'state_history': {0: bin_label},  # maps from frame id to past state predictions
            'state_prob_history': {0: bin_prob}, # maps from frame id to past state prediction prob
            'latest_frame_index': 0       # the most recent frame index that was updated
        }

    print('- Process cells in frame 0......Done~')
    # ========= 初始化统计 =========
    frame_stats = []
    num_classes = 6
    frame_idx = 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame > total_frames:
        max_frame = total_frames

    # ========= 后续帧处理 =========
    for frame_idx in tqdm(range(max_frame), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            continue

        DEBUG_PRINT(f"[frame {frame_idx}] Run segment_frame_downscaled_ds w/ cellpose_model")
        #if frame_idx<95: continue

        bboxes, masks, unique_ids, resized_frame = segment_frame_downscaled_ds(frame, cellpose_model_path, out_path,ratio=0.1, diameter=30)
        #continue
        #exit()
        #print(masks)

        matched, used_cids = match_cells_tracking(cell_info, masks, bboxes)

        for cid, info in matched.items():
            x, y, w, h = info['bbox']
            cls_id = info['class']
            cell_crop = frame[y:y+h, x:x+w]
            cell_pil = Image.fromarray(cell_crop)
            cell_tensor = transform(cell_pil).unsqueeze(0)

            with torch.no_grad():
                # 5 is ISC
                if cls_id == 5 and frame_idx>80:
                    bin_output = binary_model_F(cell_tensor.to(device))
                    bin_probs = 1 - torch.softmax(bin_output, dim=1)
                    bin_label = torch.argmax(bin_probs, dim=1).item()
                    bin_prob = bin_probs[0, bin_label].item()
                elif cls_id == 5 and frame_idx < 80:
                    bin_output = binary_model_Fb(cell_tensor.to(device))
                    bin_probs = 1 - torch.softmax(bin_output, dim=1)
                    bin_label = torch.argmax(bin_probs, dim=1).item()
                    bin_prob = bin_probs[0, bin_label].item()
                elif cls_id == 2 and frame_idx<80:
                    bin_output = binary_model_C(cell_tensor.to(device))
                    bin_probs = 1 - torch.softmax(bin_output, dim=1)
                    bin_label = torch.argmax(bin_probs, dim=1).item()
                    bin_prob = bin_probs[0, bin_label].item()
                elif cls_id == 3 and frame_idx<80:
                    bin_output = binary_model_D(cell_tensor.to(device))
                    bin_probs = 1 - torch.softmax(bin_output, dim=1)
                    bin_label = torch.argmax(bin_probs, dim=1).item()
                    bin_prob = bin_probs[0, bin_label].item()
                else:
                    bin_output = binary_model(cell_tensor.to(device))
                    bin_probs = torch.softmax(bin_output, dim=1)
                    bin_label = torch.argmax(bin_probs, dim=1).item()
                    bin_prob = bin_probs[0, bin_label].item()

            cell_info[cid]["bbox"][frame_idx] = (x, y, w, h)
            cell_info[cid]["state_history"][frame_idx] = bin_label
            cell_info[cid]["state_prob_history"][frame_idx] = bin_prob
            cell_info[cid]["latest_frame_index"] = frame_idx

    #---------------------------------------------


    # ================ Remove False Positives in bin_labels =========
    remove_bin_label_false_positives(cell_info)

    # ================ Collect Results =================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame back to zero    
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (W, H))

    for frame_index in range(max_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_skip != 0:
            continue        

        print(f"Collecting Results for frame {frame_index}......")        
        if frame_index == 0:
            out.write(frame)
            
        frame_record = {'Frame': frame_index}       
        class_counts = defaultdict(lambda: {'total': 0, 'state_1': 0})
        
        annotated_frame = frame.copy()
        for cid, info in cell_info.items():
            if frame_index in info['bbox']:
                x, y, w, h = info['bbox'][frame_index]
                bin_label = info['state_history'][frame_index]
                color = BLUE if bin_label == LABEL_UNCHANGED else RED  
                text = f"[{cid}] | C{info['class']} ({info['class_prob']:.2f})"
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated_frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)
                
                cls_id = info["class"]
                class_counts[cls_id]['total'] += 1
                if bin_label == LABEL_UNCHANGED:
                    class_counts[cls_id]['state_1'] += 1

        # Save every 10 frames
        # if frame_index % 10 == 0:
        #     DEBUG_PRINT(f"Save annotated frame png for frame index {frame_index}")
        #     cv2.imwrite(out_path+f"/frame_{frame_index}_annotated.png", annotated_frame)
                
        # Write annotated frame in output video
        out.write(annotated_frame)

        # 计算比例
        for cls_id in range(num_classes):
            total = class_counts[cls_id]['total']
            pos = class_counts[cls_id]['state_1']
            ratio = 1 - pos / total if total > 0 else 0
            frame_record[f'Class_{cls_id}'] = round(ratio, 4)
            frame_record[f'Class_{cls_id}_total'] =total
            frame_record[f'Class_{cls_id}_pos'] = pos
        frame_stats.append(frame_record)
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    print("Finished，results saved to：", output_video_path)

    # 统计每类细胞数量
    # 统计
    class_counts = Counter([info['class'] for info in cell_info.values()])
    labels = [f'{dname[i]}' for i in range(6)]
    sizes = [class_counts.get(i, 0) for i in range(6)]
    colors = plt.get_cmap('tab10').colors[:6]
    total = sum(sizes)
    percentages = [s / total * 100 for s in sizes]

    # 标签内容（供图例用）
    legend_labels = [f'{labels[i]}: {sizes[i]} ({percentages[i]:.1f}%)' for i in range(6)]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(
        sizes,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.5)
    )

    # 图例替代 annotate
    ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10)
    ax.set_title("Class Distribution in Frame 0", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path+"/frame0_class_pie.png", dpi=300)
    plt.close()

    df = pd.DataFrame(frame_stats)
    df['FrameIndex'] = range(len(df))
    df.to_csv(out_path+'/state_ratio_report.csv', index=False)

    # Old
    # plt.figure(figsize=(100, 6))
    # for cls_id in range(num_classes):
    #     #plt.plot(df['Frame'], df[f'Class_{cls_id}'], label=f'Class {cls_id}')
    #     plt.plot(df['FrameIndex'], df[f'Class_{cls_id}'], label=f'Class {cls_id}')
    # plt.xlabel('Frame')
    # plt.ylabel('Proportion of State 1')
    # plt.title('Proportion of State 1 per Class Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('output/state_ratio_plot.png', dpi=300)
    # plt.close()

    # New
    time_sec = df['FrameIndex'] * frame_skip / fps
    plt.figure(figsize=(10, 6))
    for cls_id in range(num_classes):
        y_percent = df[f'Class_{cls_id}'] * 100
        count = class_counts.get(cls_id, 0)
        #plt.plot(df['FrameIndex'], df[f'Class_{cls_id}'], label=f'Class {cls_id}')
        plt.plot(time_sec, y_percent, label=f'{dname[cls_id]} ({count} cells)')
    #plt.xlabel(f'Frame Index (every {frame_skip} frames)')
    plt.xlabel('Time (s)')
    plt.ylabel('Sickled fraction (%)')
    plt.ylim(0,100)
    #plt.title('Proportion of State 1 per Class Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path+'/state_ratio_plot.png', dpi=300)
    plt.close()
    print("Finished - generate csv and figure report.")

    plot_total_binary_ratio(df,out_path+'/state_ratio_plot_binary.png',frame_skip,fps)
    #exit()
    return cell_info,df,Counter([info['class'] for info in cell_info.values()])



#video_paths = ['demo1.mov', 'demo2.mov', 'demo3.mov']
#all_out='output_comp_filt'
#output=['out1.avi','out2.avi','out3.avi']

#os.makedirs(all_out, exist_ok=True)
#out_path=[all_out+'/video1',all_out+'/video2',all_out+'/video3']

all_stats = []
all_class_counts = Counter()

for idx, video_path in enumerate(video_paths):
    os.makedirs(out_path[idx], exist_ok=True)
    cell_info, df, class_count=process_video(video_path=video_path, out_path=out_path[idx], video_id=f"V{idx+1}", output_video_path=out_path[idx]+'/'+output[idx],six_class_model=six_class_model,binary_model=binary_model,feature_extractor=feature_extractor,transform=transform,frame_skip=frame_skip,max_frame=max_frame,fps=fps)

    all_stats.append(df)
    all_class_counts.update(class_count)

    # Save intermediate results
    save_intermediate_results(cell_info, df, out_path[idx])


# === 合并比例曲线 ===
combined_df = pd.concat(all_stats, ignore_index=True)
columns_to_sum = combined_df.columns.difference(['Frame', 'FrameIndex'])
summed = combined_df.groupby('FrameIndex')[columns_to_sum].sum().reset_index()
#print(combined_df)

# 2. 取每个 FrameIndex 对应的唯一 Frame 值（假设是相同的）
frame_map = combined_df.groupby('FrameIndex')['Frame'].first().reset_index()

# 3. 合并 Frame 列回来
final_df = pd.merge(summed, frame_map, on='FrameIndex')

# 4. 可选：把列顺序调一下（Frame 放在前面）
cols = ['FrameIndex', 'Frame'] + [col for col in final_df.columns if col not in ['FrameIndex', 'Frame']]
final_df = final_df[cols]
#print(final_df)
for cls_id in range(6):
    total_col = f'Class_{cls_id}_total'
    pos_col = f'Class_{cls_id}_pos'
    ratio_col = f'Class_{cls_id}'
    final_df[ratio_col] =np.where(final_df[total_col] > 0, 1 - final_df[pos_col] / final_df[total_col],0)

final_df.to_csv(all_out+'/state_ratio_report.csv', index=False)
#exit()

# === 画折线图 ===
time_sec = final_df['FrameIndex'] * frame_skip / fps
plt.figure(figsize=(10, 6))
for cls_id in range(6):
    y_percent = final_df[f'Class_{cls_id}'] * 100  # 转换为百分比
    count = all_class_counts.get(cls_id, 0)  # 获取对应的数量
    label = f"{dname[cls_id]} ({count} cells)"
    plt.plot(time_sec, y_percent, label=label)
plt.xlabel('Time (s)')
plt.ylabel('Sickled fraction (%)')
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(all_out+'/combined_state_ratio_plot.png', dpi=300)
plt.close()

plot_total_binary_ratio(final_df,all_out+'/state_ratio_plot_binary.png',frame_skip,fps)

# === 画合并饼图 ===
labels = [f'{dname[i]}' for i in range(6)]
sizes = [all_class_counts.get(i, 0) for i in range(6)]
colors = plt.get_cmap('tab10').colors[:6]
total = sum(sizes)
percentages = [s / total * 100 for s in sizes]
legend_labels = [f'{labels[i]}: {sizes[i]} ({percentages[i]:.1f}%)' for i in range(6)]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, _ = ax.pie(
    sizes,
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.5)
)
ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10)
ax.set_title("Total Class Distribution in Frame 0 Across All Videos", fontsize=14)
plt.tight_layout()
plt.savefig(all_out+"/combined_frame0_class_pie.png", dpi=300)
plt.close()
