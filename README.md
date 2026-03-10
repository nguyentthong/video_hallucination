# Video hallucination detection

## Download model

```shell
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./weights/qwen3_vl_8b_inst
```

## Download data

```shell
cd raw_data
curl -O https://link/to/file.mp4
```

## Setup

```shell
sudo apt update
sudo apt install -y libgl1
uv sync
source .venv/bin/activate
```