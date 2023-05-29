# скрипт для обработки сырых данных - скана цифр
from PIL import Image, ImageOps
import cv2
import numpy as np
from skimage.util import random_noise
from configs import configs


# удалить черные границы после первого кропа
def crop_borders(img: Image, color="white", border=(0, 0, 16, 20)):
    img = img.crop((3, 3, img.width - 16, img.height - 20))
    img = ImageOps.expand(img, border=border, fill=color)
    return img


# поворот на угол
def rotate_img(img: Image, name: str, angle: int):
    rname = "RL" if angle > 0 else "RR"
    M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1)
    wimg = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255)
    )
    cv2.imwrite(f"{name}-{rname}.jpg", wimg)
    # print(f"File {name}-{rname} was succesfully written.")
    return wimg


# добавить шум
def noise_img(img: Image, name: str):
    n_img = random_noise(img, mode="s&p", amount=0.008, salt_vs_pepper=0.5)
    n_img = np.array(255 * n_img, dtype=np.uint8)
    cv2.imwrite(f"{name}-N.jpg", n_img)
    # print(f"File {name}-N was succesfully written.")


# TODO: different small function
def image_augmentation(name: str):
    img = cv2.imread(f"{name}.jpg", cv2.IMREAD_GRAYSCALE)
    noise_img(img, name)

    rl = rotate_img(img, name, 20)
    noise_img(rl, name + "-RL")

    rr = rotate_img(img, name, -20)
    noise_img(rr, name + "-RR")

names = [
    "0.jpg",
    "1.jpg",
    "2.jpg",
    "3.jpg",
    "4.jpg",
    "5.jpg",
    "6.jpg",
    "7.jpg",
    "8.jpg",
    "9.jpg",
]

for dn in names:
    digit_name = dn
    config = configs[f"{digit_name[0]}"]
    image = Image.open(f"digits\\{digit_name}")

    cell_width = 176
    cell_height = 134

    num_rows = 46  # x
    num_cols = 25  # y

    y_shift = 0

    for row in range(num_rows):
        for col in range(num_cols):

            extra_x = 6 + 13 * col - config.xrowmod * row  # сдвиг вправо
            extra_y = 6 + config.ycolmod * col + row  # сдвиг вниз

            x0 = config.start_x + col * cell_width + extra_x
            y0 = config.start_y + row * cell_height + extra_y + y_shift

            x1 = x0 + cell_width
            y1 = y0 + cell_height

            cell_image = image.crop((x0, y0, x1, y1))
            cell_image = cell_image.convert("L")
            cell_data = list(cell_image.getdata())
            cell_data = [int(pixel < 160) for pixel in cell_data] # 128 or 160?

            filename = f"crop\\{digit_name[0]}\\{digit_name[0]}_{row}-{col}"

            cell_image = crop_borders(cell_image)

            # начальные пропорции - 1.31343283582 (176x134)
            # 30x22
            cell_image.thumbnail((30, int(30 / 1.31343283582)))  # 30x22
            cell_image.save(f"{filename}.jpg")

            # print(f"File {filename} was succesfully written.")

            # увеличиваем датасет
            image_augmentation(filename)
            print(f"File {filename}: OK")

        y_shift += 11.8
