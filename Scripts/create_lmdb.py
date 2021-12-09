import sys
import os.path
import glob
import pickle
import lmdb
import cv2
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

# configurations

# train_HR_folder = '/content/drive/MyDrive/SR_Code/Training/codes/data/train_HR/*.png'  # glob matching pattern
# train_HR_save_path = '/content/drive/MyDrive/SR_Code/Training/codes/lmdb_dataset/train_HR_lmdb_n'  # must end with .lmdb
# train_LR_folder = '/content/drive/MyDrive/SR_Code/Training/codes/data/train_LR/*.png'  # glob matching pattern
# train_LR_save_path = '/content/drive/MyDrive/SR_Code/Training/codes/lmdb_dataset/train_LR_lmdb_n'  # must end with .lmdb
# val_HR_folder = '/content/drive/MyDrive/SR_Code/Training/codes/data/val_HR/*.png'  # glob matching pattern
# val_HR_save_path = '/content/drive/MyDrive/SR_Code/Training/codes/lmdb_dataset/val_HR_lmdb_n'  # must end with .lmdb
# val_LR_folder = '/content/drive/MyDrive/SR_Code/Training/codes/data/val_LR/*.png'  # glob matching pattern
# val_LR_save_path = '/content/drive/MyDrive/SR_Code/Training/codes/lmdb_dataset/val_LR_lmdb_n'  # must end with .lmdb

input_dir = sys.argv[1]
save_path = sys.argv[2]

print("input_dir: ", input_dir)

fl_dict = {input_dir:save_path}

for img_folder, lmdb_save_path in fl_dict.items():
    if os.path.exists(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)
    os.mkdir(lmdb_save_path)

    img_list = sorted(glob.glob(img_folder))
    dataset = []
    data_size = 0
    print('Read images...')
    pbar = ProgressBar(len(img_list))

    for i, v in enumerate(img_list):
        pbar.update('Read {}'.format(v))
        img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        dataset.append(img)
        data_size += img.nbytes
        print("read end")
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

    pbar = ProgressBar(len(img_list))
    with env.begin(write=True) as txn:  # txn is a Transaction object
        for i, v in enumerate(img_list):
            pbar.update('Write {}'.format(v))
            base_name = os.path.splitext(os.path.basename(v))[0]
            key = base_name.encode('ascii')
            data = dataset[i]
            if dataset[i].ndim == 2:
                H, W = dataset[i].shape
                C = 1
            else:
                H, W, C = dataset[i].shape
            meta_key = (base_name + '.meta').encode('ascii')
            meta = '{:d}, {:d}, {:d}'.format(H, W, C)
            # The encode is only essential in Python 3
            txn.put(key, data)
            txn.put(meta_key, meta.encode('ascii'))
    print('Finish writing lmdb.' )

    # create keys cache
    keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
    env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        print('Create lmdb keys cache: {}'.format(keys_cache_file))
        keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')
