# Thunder Compute VM Bootstrap

Steps to get a fresh Thunder Compute instance ready for training.

---

## 1. Clone repo

```bash
git clone https://github.com/cdrue3/TF-CLIP-quantum
cd TF-CLIP-quantum
```

## 2. Install dependencies

```bash
pip3 install pennylane timm einops yacs ftfy regex tqdm tensorboard gdown \
    opencv-python-headless transformers open_clip_torch
```

## 3. Install rclone

```bash
curl -s https://rclone.org/install.sh | sudo bash
```

## 4. Authenticate rclone with Google Drive

```bash
rclone authorize "drive" --auth-no-open-browser
```

- Terminal prints a URL like `http://127.0.0.1:53682/auth?state=XXX`
- **Do NOT use VS Code "Open in Browser"** — copy the full URL and paste it directly into your browser address bar (VS Code port-forwards localhost:53682)
- Authorize with your Google account
- Terminal prints a JSON token — copy it

Create the rclone config:

```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'EOF'
[gdrive]
type = drive
token = PASTE_JSON_TOKEN_HERE
EOF
```

Verify:

```bash
rclone lsd gdrive: | head -5
```

## 5. Transfer dataset

**Option A — tar via SCP from local machine (fastest)**

On your local machine:
```bash
tar -cf ag_vpreid.tar /path/to/AG-VPReID/
scp ag_vpreid.tar ubuntu@<vm-ip>:~/TF-CLIP-quantum/DATA/
```

On the VM:
```bash
mkdir -p ~/TF-CLIP-quantum/DATA
tar -xf ~/TF-CLIP-quantum/DATA/ag_vpreid.tar -C ~/TF-CLIP-quantum/DATA/
rm ~/TF-CLIP-quantum/DATA/ag_vpreid.tar
```

**Option B — rclone from Google Drive (slow for large datasets, ~100k small files)**

```bash
mkdir -p ~/TF-CLIP-quantum/DATA/AG-VPReID
rclone copy "gdrive:AG-VPReID" ~/TF-CLIP-quantum/DATA/AG-VPReID/ \
    --transfers 16 --checkers 16 --drive-chunk-size 256M --progress
```

## 6. Git push setup (optional)

```bash
git config user.email "connordruett@hotmail.com"
git config user.name "Connor Druett"
# Generate a PAT at github.com → Settings → Developer settings → PATs
# Give it Contents: read+write on TF-CLIP-quantum
git remote set-url origin https://YOUR_PAT@github.com/cdrue3/TF-CLIP-quantum
# Reset after pushing so token isn't stored in config
git remote set-url origin https://github.com/cdrue3/TF-CLIP-quantum
```

---

## Notes

- No conda — use `python` / `pip3` directly
- GPU: A6000 (49GB) or T4 depending on instance
- Training uses `DATA/subset_250/` for quick runs, full `DATA/AG-VPReID/` for 80ep runs
- Do NOT snapshot while training is actively writing logs — it will stall
