from tensorflow import keras
from keras.utils import load_img, img_to_array
import numpy as np
import os, random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle


matplotlib.use("TkAgg")

subfolders = [f.path for f in os.scandir("crop/") if f.is_dir()]
rdict = {}
model = keras.models.load_model("CNN-1.keras")

for subf in subfolders:
    for i in range(2):
        choice = random.choice(os.listdir(subf))
        img = load_img(f"{subf}/{choice}", target_size=(29, 22), color_mode="grayscale")
        img_arr = (img_to_array(img) / 255.0).reshape(-1)
        img_arr = img_arr.reshape(1, 29, 22, 1)
        rdict[choice] = img_arr

fig, axs = plt.subplots(4, 5, figsize=(10, 4))

for i, (key, value) in enumerate(rdict.items()):
    prediction = model.predict(value)
    prediction = prediction.tolist()[0]
    prediction = [f"{j} - " + str(round(i, 5)) for j, i in enumerate(prediction) if i > 0.001]

    max_index = np.argmax(prediction)
    
    img = load_img(f"crop/{key[0]}/{key}", target_size=(29, 22), color_mode="grayscale")
    ax = axs[i//5, i%5]
    ax.imshow(img, cmap="gray", extent=[0, 1, 0, 1])
    ax.axis("off")
    ax.set_title("\n".join(prediction), fontsize=12)
    rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.show()