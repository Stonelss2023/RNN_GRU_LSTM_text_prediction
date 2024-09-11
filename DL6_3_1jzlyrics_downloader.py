import urllib.request
import os
import zipfile
import io

# 全局变量来跟踪数据是否已加载
_data_loaded = False
_cached_data = None

def download_data(data_dir='data', filename='jaychou_lyrics.txt'):
    global _data_loaded, _cached_data#声明全局变量，允许在函数内部修改它们

    if _data_loaded:
        return _cached_data#如果数据已加载，直接返回缓存数据
    
    #构建本地文件完整路径
    os.makedirs(data_dir, exist_ok=True)
    local_file = os.path.join(data_dir, filename)

    #如果本地文件存在→读取内容，更新缓存状态并加载状态，返回数据
    if os.path.exists(local_file):
        print("Loading data from local file...")
        with open(local_file, 'r', encoding='utf-8') as f:
            _cached_data = f.read()
        _data_loaded = True
        return _cached_data

    url = "https://d2l-data.s3-accelerate.amazonaws.com/jaychou_lyrics.txt.zip"
    
    print(f"Downloading from: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()
        
        print("File downloaded successfully.")
        
        #解压下载的ZIP文件，读取并解码内容
        with zipfile.ZipFile(io.BytesIO(content)) as zin:
            with zin.open(filename) as f:
                corpus_chars = f.read().decode('utf-8')
        
        #解压内容保存到本地文件
        with open(local_file, 'w', encoding='utf-8') as f:
            f.write(corpus_chars)
        
        #更新缓存和加载状态，返回数据
        print("Data saved to local file.")
        _cached_data = corpus_chars
        _data_loaded = True
        return corpus_chars

    #如果下载过程中出现错误，打印错误信息并抛出异常
    except Exception as e:
        print(f"Failed to download the file. Error: {e}")
        raise

