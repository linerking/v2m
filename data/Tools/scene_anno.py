import os
import json
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path, output_path, threshold=30.0):
    # 创建视频管理器
    video_manager = VideoManager([video_path])
    
    # 创建场景管理器
    scene_manager = SceneManager()
    
    # 添加内容检测器
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # 启动视频管理器
    video_manager.set_downscale_factor()
    video_manager.start()
    
    # 处理视频并检测场景
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # 获取检测到的场景列表
    scene_list = scene_manager.get_scene_list()
    
    # 打印检测到的场景
    print(f'{len(scene_list)} scenes detected:')
    with open(output_path, 'w') as f:
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            f.write(f'{start_time:.2f}  {end_time:.2f}\n')
            print(f'{start_time:.2f}  {end_time:.2f}')
    
    # 释放视频管理器资源
    video_manager.release()

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    raw_video_path = config["raw_video_path"]
    scene_anno_path = config["scene_anno_path"]
    
    os.makedirs(scene_anno_path, exist_ok=True)
    
    for filename in os.listdir(raw_video_path):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(raw_video_path, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(scene_anno_path, output_filename)
            detect_scenes(video_path, output_path)