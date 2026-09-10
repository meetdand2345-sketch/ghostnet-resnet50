from pathlib import Path
from huggingface_hub import snapshot_download
def main():
    out=Path('data/raw'); out.mkdir(parents=True,exist_ok=True)
    print('Downloading rehan9599/drishti-sss with reduced concurrency...')
    snapshot_download(repo_id='rehan9599/drishti-sss',repo_type='dataset',local_dir=str(out),max_workers=2,resume_download=True)
    print('Download complete:',out.resolve())
if __name__=='__main__': main()
