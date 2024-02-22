import cv2
import os
import tqdm


def AHE(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # image = image.astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # output = output.astype(np.float32)
    return image


root = 'D:/Side/2024_Sejoong_Jaywalking/DB/detection/test/'
output = f'{root}/clahe_images'
os.makedirs(output, exist_ok=True)

image_names = os.listdir(f'{root}/images')

for name in tqdm.tqdm(image_names):
    img = cv2.imread(f'{root}/images/{name}')
    img = AHE(img)
    cv2.imwrite(f'{output}/{name}', img)
