import os, sys
import cv2
import numpy as np
from PIL import Image

def augment(img, hflip=True, rot=True):
    # horizontal flip OR rotate
    res_imgs = [
      img.transpose(Image.FLIP_LEFT_RIGHT),
      img.transpose(Image.FLIP_TOP_BOTTOM),
      img.transpose(Image.ROTATE_90),
      img.transpose(Image.ROTATE_180),
      img.transpose(Image.ROTATE_270)
    ]

    return res_imgs

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python %s INPUT_DIR OUTPUT_DIR" % sys.argv[0])
        sys.exit(-1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    items = os.listdir(input_dir)
    index = 1
    for item in items:
      if not item.endswith('.png'):
        continue
      in_file = os.path.join(input_dir,item)
      in_img = Image.open(in_file)
      out_imgs = augment(in_img)
      out_file = os.path.join(output_dir, "%06d.png" % index)
      index += 1
      in_img.save(out_file)
      for img in out_imgs:
        out_file = os.path.join(output_dir, "%06d.png" % index)
        index += 1
        img.save(out_file)