from easyocr import easyocr
import numpy as np

from torchvision import transforms

from PIL import Image

# img_path = "data_in/beispiel_text.jpg"
# img_path = "data_in/beispiel_text.png"
# img_path = "data_in/code.png"
img_path = "data_in/input.png"


def read_text(image_name, model_name, in_line=True):
    # Read the data
    text = model_name.readtext(image_name, detail = 0, paragraph=in_line)

    # Join texts writing each text in new line
    return '\n'.join(text)


img = Image.open(img_path)
dd = np.asarray(img)
convert_tensor = transforms.ToTensor()
img_tensor = convert_tensor(img)
np_arr = img_tensor.cpu().detach().numpy()

# reader_en_ch = easyocr.Reader(['en', 'ch_sim'])
reader_en_de = easyocr.Reader(['en', 'de'])

ch_text = read_text(img_path, reader_en_de)
print(ch_text)
