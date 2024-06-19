from clip_inference import inference
from glob import glob
from PIL import Image
import pandas as pd


ground_truth = pd.read_csv("Data_theme_deforestation_with_filenames.csv")
filenames = list(ground_truth["ImageID"])
print(filenames)

labels = [
    "Coastal Seas", 
    "Deserts",
    "Forests",
    "Freshwater",
    "Grasslands",
    "Mountains",
    "Open Ocean",
    "Polar",
    "Rivers",
    "Rural",
    "Urban",
    "Wetlands"
]

gt_simple = []
for i, hab in enumerate(ground_truth["Habitat"]):
    hab = hab.split(";")[0].strip()
    try:
        gt_simple.append([int(ground_truth["ImageID"][i]), labels.index(hab)])
    except ValueError as e:
        pass

Y = [gt[1] for gt in gt_simple]
print(Y)
X = []
Y = []
real_filenames = glob("downloaded_images_deforestation/*.jpg")
for filename, gt in gt_simple:
    f = f"downloaded_images_deforestation/{str(filename)}.jpg"
    if f in real_filenames:
        image = Image.open(f"downloaded_images_deforestation/{str(filename)}.jpg")
        preds = inference(image, labs_to_use="habitat", threshold=0.01)
        X.append(preds[0][1])
        Y.append(gt)


print(X)
print(Y)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(Y, X)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Show the plot
plt.show()

# #print(filenames)
# #print(ground_truth.describe())

