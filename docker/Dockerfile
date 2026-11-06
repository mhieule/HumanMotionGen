# MDM (Tevet et al.) for sampling on an RTX 3060.
# Replaces the repo's PyTorch 1.7.1/CUDA 11.0 conda env (no sm_86 kernels) with
# pip wheels of PyTorch 1.13.1+cu117, which bundle their own CUDA libraries.
# Only the host's NVIDIA driver is needed; Docker passes the GPU in with --gpus.
FROM ubuntu:20.04

ARG MDM_COMMIT=ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/mdm \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git ffmpeg unzip ca-certificates \
        build-essential libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# numpy < 1.24: MDM and chumpy still use np.float / np.int / np.bool.
# matplotlib 3.1.3: newer versions break ax.grid(b=False) in plot_script.py.
# moviepy 1.0.3: the code imports moviepy.editor, removed in moviepy 2.
RUN pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.1.3 \
        scikit-learn==1.0.2 h5py==3.7.0 tqdm==4.64.1 trimesh==3.15.3 joblib==1.1.0 \
        spacy==3.3.1 smplx==0.1.28 ftfy==6.1.1 regex==2022.9.13 \
        moviepy==1.0.3 imageio==2.22.2 "gdown>=5.2" \
    && pip install --no-build-isolation chumpy==0.70 \
    && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl \
    && pip install git+https://github.com/openai/CLIP.git \
    && pip install typing-extensions==4.1.1  # newer ones break spaCy's pydantic 1.8 on import

RUN git clone https://github.com/GuyTevet/motion-diffusion-model.git /mdm \
    && git -C /mdm checkout ${MDM_COMMIT}

# Bake the CLIP ViT-B/32 weights into the image, and leave HOME writable for
# whichever host uid the container runs as (so mounted outputs stay yours).
RUN python -c "import clip; clip.load('ViT-B/32', device='cpu')" \
    && chmod -R a+rwX /home/mdm

# eval/eval_humanml.py imports train.train_platforms (needs wandb); train/training_loop.py needs blobfile.
RUN pip install wandb==0.15.12 blobfile==2.1.1

# Train and eval cache each processed split as ./dataset/t2m_<split>.npy. Point it into
# the mounted save/ folder, so the host uid can write it and it survives --rm.
RUN for s in train val test; do ln -s /mdm/save/.cache_t2m_$s.npy /mdm/dataset/t2m_$s.npy; done

WORKDIR /mdm
CMD ["bash"]
