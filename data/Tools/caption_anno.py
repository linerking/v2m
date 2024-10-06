import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from v2m_models.BLIP.models.blip import blip_decoder
from PIL import Image
import cv2
import json
from pathlib import Path

# 将项目根目录添加到Python路径


def load_image(image, image_size, device):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 确保图像是 RGB 格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = transform(image).unsqueeze(0).to(device)
    return image

def caption_anno(video_path, start_time, end_time, model, device, image_size):
    if not os.path.isfile(video_path):
        raise ValueError(f"Invalid video file path: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    duration = end_time - start_time
    captions = []
    
    for i in range(10):  # Process 5 frames
        frame_time = start_time + (i + 1) * (duration / 6)
        frame_number = int(frame_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at time: {frame_time}")
            continue
        
        print(f"Processing frame at time: {frame_time}")
        
        # 将 OpenCV 的 BGR 格式转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = load_image(image, image_size, device)
        
        with torch.no_grad():
            caption = model.generate(image, sample=True, num_beams=30, max_length=50, min_length=20,repetition_penalty=1.5, top_p=0.5)
        captions.append(caption[0])
    
    cap.release()
    return captions

def process_scene_file(scene_file_path, raw_video_dir, model, device, image_size):
    scene_captions = []
    video_name = os.path.basename(scene_file_path).split('.')[0]
    video_path = os.path.join(raw_video_dir, f"{video_name}.mp4")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found.")
    
    print(f"Processing video: {video_path}")
    
    with open(scene_file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"Warning: Line {line_number} in {scene_file_path} is not in the correct format. Skipping.")
                continue
            start_time, end_time = parts
            try:
                print(f"Processing scene {line_number}: {start_time} - {end_time}")
                scene_captions_list = caption_anno(video_path, float(start_time), float(end_time), model, device, image_size)
                scene_captions.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "captions": scene_captions_list
                })
                print(f"Captions generated for scene {line_number}")
            except Exception as e:
                print(f"Error processing line {line_number} in {scene_file_path}: {str(e)}")
                continue
    
    return video_name, scene_captions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    raw_video_path = Path(config['raw_video_path'])
    scene_anno_path = Path(config['scene_anno_path'])
    caption_anno_path = Path(config['caption_anno_path'])
    
    scene_file = scene_anno_path / '0.txt'
    print(f"Processing scene file: {scene_file}")
    
    try:
        video_name, scene_captions = process_scene_file(scene_file, raw_video_path, model, device, image_size)
        if not scene_captions:
            print(f"Warning: No valid scenes found in {scene_file}")
        else:
            output_file = caption_anno_path / f'{video_name}_captions.json'
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(scene_captions, f, indent=4)
            print(f"Captions for {video_name} saved to {output_file}")
    except FileNotFoundError as e:
        print(f"Error: File not found. {str(e)}")
    except Exception as e:
        print(f"Error processing {scene_file}: {str(e)}")
        import traceback
        traceback.print_exc()