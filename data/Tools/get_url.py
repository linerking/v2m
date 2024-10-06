#爬取youtube一个视频合辑中全部视频的url
import yt_dlp as youtube_dl

def get_youtube_video_urls_with_index(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
    }
    
    video_urls_with_index = []
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)
        
        if 'entries' in result:
            for index, entry in enumerate(result['entries']):
                video_urls_with_index.append((index, entry['url']))
        else:
            video_urls_with_index.append((0, result['url']))
    
    return video_urls_with_index

def save_urls_to_file(video_urls_with_index, start_index, end_index, file_path):
    with open(file_path, 'w') as file:
        for index, url in video_urls_with_index:
            if start_index <= index <= end_index:
                file.write(f"Index: {index}, URL: {url}\n")

# Example usage
if __name__ == '__main__':
    playlist_url = 'https://www.youtube.com/playlist?list=PLPA4SrN8zUHr0rIDH_WdO_8rwPed_icgK'
    video_urls_with_index = get_youtube_video_urls_with_index(playlist_url)
    
    start_index = 0  # 指定起始索引
    end_index = 10   # 指定结束索引
    file_path = '/home/yihan/v2m/data/url.txt'
    
    save_urls_to_file(video_urls_with_index, start_index, end_index, file_path)
    print(f"URLs from index {start_index} to {end_index} have been saved to {file_path}")