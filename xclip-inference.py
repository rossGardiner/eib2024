import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

np.random.seed(0)

text = [
    "a photo in Coastal Seas habitat",
    "a photo in Deserts habitat",
    "a photo in Forests habitat",
    "a photo in Freshwater habitat",
    "a photo in Grasslands habitat",
    "a photo in Mountains habitat",
    "a photo in Open Ocean habitat",
    "a photo in Polar habitat",
    "a photo in Rivers habitat",
    "a photo in Rural habitat",
    "a photo in Urban habitat",
    "a photo in Wetlands habitat"
]

#text=["playing sports", "eating spaghetti", "go shopping"]

# text = [
#     "a photo with Biodiversity theme",
#     "a photo with Climate theme",
#     "a photo with Consumption theme", 
#     "a photo with Deforestation theme", 
#     "a photo with Energy theme",
#     "a photo with Extreme Weather theme",
#     "a photo with Food theme",
#     "a photo with Human Health theme",
#     "a photo with Land Management theme",
#     "a photo with Natural Disasters theme",
#     "a photo with Nature and Wildlife theme",
#     "a photo with Plastics theme",
#     "a photo with Pollution theme", 
#     "a photo with Waste theme",
#     "a photo with Water theme",
#     "a photo with Sustainable Future theme",
#     "a photo with Technology theme"
# ]

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = "PND104_20230811_C1558_HD.mp4"
container = av.open(file_path)

# sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

inputs = processor(
    text=text,
    videos=list(video),
    return_tensors="pt",
    padding=True,
)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)