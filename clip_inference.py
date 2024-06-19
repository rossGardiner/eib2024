from PIL import Image

import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://cdnt3m-a.akamaihd.net/tem/warehouse/1DD/CC1/1DDCC1_PC3TV2Q4VE_lt.jpg"

image = Image.open(requests.get(url, stream=True).raw)


def inference(image, labs_to_use="habitat", threshold=0.5):
    if labs_to_use == "habitat":
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
    elif labs_to_use == "theme":
        text = [
            "a photo with Biodiversity theme",
            "a photo with Climate theme",
            "a photo with Consumption theme", 
            "a photo with Deforestation theme", 
            "a photo with Energy theme",
            "a photo with Extreme Weather theme",
            "a photo with Food theme",
            "a photo with Human Health theme",
            "a photo with Land Management theme",
            "a photo with Natural Disasters theme",
            "a photo with Nature and Wildlife theme",
            "a photo with Plastics theme",
            "a photo with Pollution theme", 
            "a photo with Waste theme",
            "a photo with Water theme",
            "a photo with Sustainable Future theme",
            "a photo with Technology theme"
        ]

#text=["a photo of a cat", "a photo of a dog"]

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

    probs = logits_per_image.softmax(dim=1)  
    probs = probs.tolist()
    pairs = zip(probs[0], text)
    print(probs[0])
    filtered_pairs = []
    for pair in pairs:
        if pair[0] >= threshold:
            filtered_pairs.append((pair[0], text.index(pair[1])))
    filtered_pairs.sort(key=lambda x: x[0], reverse=True)
    return filtered_pairs

# p = inference(image, "habitat", threshold=0.01)
# print(p)