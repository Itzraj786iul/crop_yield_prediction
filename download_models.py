import os
import requests
import zipfile
from tqdm import tqdm
import hashlib

# URLs for your model files (replace with your actual URLs)
MODEL_URLS = {
    'crop_yield_model.pkl': 'https://drive.google.com/file/d/1-X0p-2SRjT9uCavtrVawC_tCGOGX3p62/view?usp=sharing',
    'base_scaler.pkl': 'https://drive.google.com/file/d/1U2vve8wcCfySHeX5wCF9MfnUmyGag1ep/view?usp=sharing',
    'interaction_scaler.pkl': 'https://drive.google.com/file/d/1-YQoip1Mwd_cXLDeZkaydJb49BEDYnAK/view?usp=sharing',
    'label_encoders.pkl': 'https://drive.google.com/file/d/1EXME2KOfvwQkESy83o2n1vjug7Lb0qQo/view?usp=sharing',
    'crop_to_category.pkl': 'https://drive.google.com/file/d/1lqb2JgF-mph0AOHGS7U8xrTZ6in-jYfY/view?usp=sharing',
    'feature_cols.pkl': 'https://drive.google.com/file/d/1go17f1or8oeQRD24bXBoxMlpXlWUd-U0/view?usp=sharing'
}

def download_file(url, filename):
    """Download a file from a URL with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(filename, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB', unit_scale=False):
                f.write(data)
        
        print(f"Downloaded {filename} successfully")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def verify_file_hash(filename, expected_hash=None):
    """Verify file hash if provided."""
    if expected_hash is None:
        return True
    
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def download_models():
    """Download all model files."""
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(model_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"{filename} already exists, skipping download")
            continue
        
        print(f"Downloading {filename}...")
        success = download_file(url, filepath)
        
        if not success:
            print(f"Failed to download {filename}")
            return False
    
    print("All models downloaded successfully")
    return True

if __name__ == "__main__":
    download_models()