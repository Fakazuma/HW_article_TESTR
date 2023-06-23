import json
import os
from collections import defaultdict


def all_data_together(datadir, file):
    a = os.listdir(datadir)
    full = []
    for p in a:
        if p == 'train_test_full':
            continue
        d_f = os.path.join(datadir, p)
        f_path = os.path.join(d_f, file)
        with open(f_path, 'r') as f:
            json.load(f)


def change_path(file, new_path):
    with open(file, 'r') as f:
        data = json.load(f)

    new_data = []
    for el in data:
        old_path = el['file_name']
        new_path_ = os.path.join(new_path, *old_path.split(r'/')[-2:])
        el['file_name'] = new_path_
        new_data.append(el)

    dir = os.path.dirname(file)
    n_file = file.split('/')[-1].split('.')[0] + '_win.json'
    n_file = os.path.join(dir, n_file)

    with open(n_file, 'w') as f:
        json.dump(new_data, f)


class IcdarParser:
    def __init__(self, path):
        self.path = path

    def parse(self):
        for t in ['train', 'test']:
            anno = os.path.join(self.path, f'{t}_poly.json')
            im_dir = os.path.join(self.path, f'{t}_images')

            with open(anno, 'r', encoding='utf-8') as f:
                data = json.load(f)

            size = len(data['images'])
            parsed_data = []

            sampales_info = {}
            for i in range(size):
                samp_info = defaultdict(list)
                im = data['images'][i]
                samp_info['image'] = os.path.join(im_dir, im['file_name'])
                samp_info['id'] = im['id']
                sampales_info[im['id']] = samp_info

            meta = data['annotations']
            for samp in meta:
                id = samp['image_id']
                sampales_info[id]['bboxes'].append(samp['bbox'])
                sampales_info[id]['texts'].append(samp['rec'])
                x_pts = samp['polys'][0::2]
                y_pts = samp['polys'][1::2]
                new_polys = [[x, y] for x, y in zip(x_pts, y_pts)]
                sampales_info[id]['polys'].append(new_polys)

            l = list(sampales_info.values())

            new_data_path = os.path.join(self.path, f'{t}.json')
            with open(new_data_path, 'w') as f:
                json.dump(l, f)


if __name__ == '__main__':
    dir_path = r'C:\Users\Admin\PycharmProjects\TESTR\datasets\icdar2015'
    parser = IcdarParser(dir_path)
    parser.parse()