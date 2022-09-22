import os
import shutil

from tqdm import tqdm

eval_dir = './results_eval/20220922153434'

if __name__ == '__main__':
    log_path = os.path.join(eval_dir, 'log.txt')
    debug_dir = os.path.join(eval_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    with open(log_path) as rf:
        rf.readline()
        for line in tqdm(rf.readlines()):
            if line[0] == '*':
                break
            img_path, t, p = line.replace('\n', '').split(',')
            img_filename = os.path.basename(img_path)
            fn, ext = os.path.splitext(img_filename)
            gt_filename = '{}_GT{}'.format(fn, ext)
            gt_path = img_path.replace(img_filename, gt_filename)

            if not os.path.isfile(img_path):
                print('File Not Found! ({})'.format(img_path))
                raise FileNotFoundError
            if not os.path.isfile(gt_path):
                print('File Not Found! ({})'.format(gt_path))
                raise FileNotFoundError

            save_dir = os.path.join(debug_dir, 'true {} pred {}'.format(t, p))
            os.makedirs(save_dir, exist_ok=True)

            shutil.copy2(img_path, os.path.join(save_dir, img_filename))
            shutil.copy2(gt_path, os.path.join(save_dir, gt_filename))
