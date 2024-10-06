import json
import os
import yt_dlp
import logging
from datetime import datetime
import shutil

# 读取config.json文件
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

url_path = config['url_path']
raw_video_path = config['raw_video_path']
download_log_path = config['download_log_path']

# 配置yt-dlp
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'merge_output_format': 'mp4'
}

# 配置日志
logging.basicConfig(filename=download_log_path, level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 下载单个视频
def download_video(url, output_path, index):
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            ydl_opts['outtmpl'] = os.path.join(output_path, f'{index}.%(ext)s')
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            success_message = f"视频已成功下载: 索引 {index}"
            print(success_message)
            logging.info(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "index": index,
                "status": "success"
            }))
            return  # 下载成功，退出函数
        except Exception as e:
            error_message = f"下载失败: 索引 {index}, URL: {url}, 错误: {str(e)}, 尝试次数: {attempt + 1}"
            print(error_message)
            logging.error(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "index": index,
                "url": url,
                "error": str(e),
                "attempt": attempt + 1
            }))
            
            # 删除可能存在的部分下载文件
            for file in os.listdir(output_path):
                if file.startswith(f'{index}.'):
                    file_path = os.path.join(output_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            
            if attempt == max_attempts - 1:
                失败消息 = f"视频下载失败: 索引 {index}，已达到最大尝试次数"
                print(失败消息)
                logging.error(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "index": index,
                    "status": "failed",
                    "message": 失败消息
                }))

# 从url_path逐行读取并下载视频
with open(url_path, 'r') as url_file:
    for index, url in enumerate(url_file):
        url = url.strip()  # 去除行末的换行符
        if url:  # 确保url不为空
            download_video(url, raw_video_path, index)
