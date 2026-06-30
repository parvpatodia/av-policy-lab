"""Generalized nuPlan split downloader from the public AWS Open Data bucket
(no creds, --no-sign-request) via the cluster proxy. Args: <s3_key> <dest_path>.
Skips if the local file already matches the remote size (safe re-run)."""
import sys, os, threading, boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

BUCKET = "motional-nuplan"
KEY = sys.argv[1]
dest = sys.argv[2]
proxies = {}
if os.environ.get("https_proxy"):
    proxies["https"] = os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    proxies["http"] = os.environ["http_proxy"]
cfg = Config(signature_version=UNSIGNED, proxies=proxies or None,
             retries={"max_attempts": 5, "mode": "standard"}, max_pool_connections=20)
s3 = boto3.client("s3", region_name="ap-northeast-1", config=cfg)
size = s3.head_object(Bucket=BUCKET, Key=KEY)["ContentLength"]
print("remote_size_GB %.1f" % (size / 1024**3), flush=True)
if os.path.exists(dest) and os.path.getsize(dest) == size:
    print("ALREADY_COMPLETE")
    sys.exit(0)
done = [0]; last = [0]; lock = threading.Lock()
def cb(n):
    with lock:
        done[0] += n
        if done[0] - last[0] >= 5 * 1024**3:
            last[0] = done[0]
            print("progress %.1f/%.1f GB" % (done[0] / 1024**3, size / 1024**3), flush=True)
tc = TransferConfig(multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=16)
s3.download_file(BUCKET, KEY, dest, Config=tc, Callback=cb)
print("DOWNLOAD_COMPLETE", os.path.getsize(dest), flush=True)
