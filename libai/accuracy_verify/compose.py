import PIL.Image as Image
import os

IMAGES_PATH = './img_model/'

IMAGES_FORMAT = ['.png']
IMAGE_SIZE_1 = 640
IMAGE_SIZE_2 = 480
IMAGE_ROW = 2
IMAGE_COLUMN = 3
IMAGE_SAVE_PATH = './curve/compose_model.png'

image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
print(image_names)

if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_1, IMAGE_ROW * IMAGE_SIZE_2))
    
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE_1, IMAGE_SIZE_2), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_1, (y - 1) * IMAGE_SIZE_2))
    return to_image.save(IMAGE_SAVE_PATH)


image_compose()
