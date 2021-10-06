# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels


def load_depthmap(filename, img_width, img_height, max_depth):
    with open(filename, mode='rb') as f:
        data = f.read()
        num_pixel = (img_width*img_height)
        cropped_image = struct.unpack('f'*num_pixel, data[:num_pixel * 4 ])
        cropped_image = np.asarray(cropped_image).reshape(img_height, -1)
        depth_image = np.zeros((img_height, img_width), dtype=np.float32)
        depth_image[0:img_height, 0:img_width ] = cropped_image
        depth_image[depth_image == 0] = max_depth

        return depth_image


class NYUDataset(Dataset):
    def __init__(self, root, center_dir, mode,  transform=None):
        self.img_width = 640
        self.img_height = 480
        self.max_depth = 1200
        self.fx = 588.03
        self.fy = 587.07
        self.joint_num = 21
        self.world_dim = 3
        self.root = root
        self.center_dir = center_dir
        self.mode = mode
        self.transform = transform
   
        self._load()
    
    def __getitem__(self, index):
        depthmap_img = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        points = depthmap2points(depthmap_img, self.fx, self.fy)
        points = points.reshape((-1, 3))
        sample = {
            'name': self.names[index],
            'points': points,
            'refpoint': self.ref_pts[index]
        }

        if self.transform: sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return self.num_samples


    def _load(self):

        self.num_samples = 8252
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        ref_pt_file = 'test_center_uvd.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
                ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for i in range(0, 8252):
            # referece point
            splitted = ref_pt_str[file_id].split()
            if splitted[0] == 'invalid':
                print('Warning: found invalid reference frame')
                file_id += 1
                continue
            else:
                self.ref_pts[frame_id, 0] = float(splitted[0])
                self.ref_pts[frame_id, 1] = float(splitted[1])
                self.ref_pts[frame_id, 2] = float(splitted[2])

            filename = os.path.join(self.root, str(i)  + '.bin')
            self.names.append(filename)    

            frame_id += 1
            file_id += 1

