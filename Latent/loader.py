import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else: # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)

class ct_ed_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        edge_path = sorted(glob(os.path.join(saved_path, '*_edge.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]#[0:10]
            target_ = [f for f in target_path if test_patient not in f]#[0:10]
            edge_ = [f for f in edge_path if test_patient not in f]#[0:10]
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
                self.edge_ = edge_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
                self.edge_ = [np.load(f) for f in edge_]
        else: # mode =='test'
            input_ = [f for f in input_path if test_patient in f]#[0:10]
            target_ = [f for f in target_path if test_patient in f]#[0:10]
            edge_ = [f for f in edge_path if test_patient in f]#[0:10]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
                self.edge_ = edge_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
                self.edge_ = [np.load(f) for f in edge_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img,edge_img = self.input_[idx], self.target_[idx], self.edge_[idx]
        if self.load_mode == 0:
            input_img, target_img,edge_img = np.load(input_img), np.load(target_img), np.load(edge_img)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            edge_img = self.transform(edge_img)
        if self.patch_size:
            input_patches, target_patches,edge_patches = get_patch3(input_img,
                                                      target_img,
                                                      edge_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, np.concatenate((target_patches,edge_patches),axis=0))#xiecuole
        else:
            input_img=np.expand_dims(input_img,0)
            edge_img = np.expand_dims(edge_img, 0)
            target_img = np.expand_dims(target_img, 0)

            return input_img,edge_img,target_img


class ct_latent_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path_1, saved_path_2, test_patient, patch_n=None, patch_size=None,
                 transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0, 1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path_1, '*_input.npy')))
        target_path = sorted(glob(os.path.join(saved_path_1, '*_target.npy')))
        input_latent_path = sorted(glob(os.path.join(saved_path_2, '*_input.npy')))
        target_latent_path = sorted(glob(os.path.join(saved_path_2, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

            input_latent_ = [f for f in input_latent_path if test_patient not in f]
            target_latent_ = [f for f in target_latent_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_latent_ = input_latent_
                self.target_latent_ = target_latent_
            else:  # all data load
                self.input_latent_ = [np.load(f) for f in input_latent_]
                self.target_latent_ = [np.load(f) for f in target_latent_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]#[0:16]
            target_ = [f for f in target_path if test_patient in f]#[0:16]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
            input_latent_ = [f for f in input_latent_path if test_patient in f]
            target_latent_ = [f for f in target_latent_path if test_patient in f]
            # print(input_)
            # print(target_latent_)
            if load_mode == 0:
                self.input_latent_ = input_latent_
                self.target_latent_ = target_latent_
            else:
                self.input_latent_ = [np.load(f) for f in input_latent_]
                self.target_latent_ = [np.load(f) for f in target_latent_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        input_latent, target_latent = self.input_latent_[idx], self.target_latent_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)
            input_latent, target_latent = np.load(input_latent), np.load(target_latent)

        input_img = np.expand_dims(input_img, 0)
        target_img = np.expand_dims(target_img, 0)
        return (input_img, target_img,input_latent, target_latent)

def get_patch3(full_input_img, full_target_img,full_edge_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    patch_edge_imgs=[]
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_edge_img = full_edge_img[top:top + new_h, left:left + new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        patch_edge_imgs.append(patch_edge_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs), np.array(patch_edge_imgs)

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode='train', load_mode=0,
               saved_path=None, test_patient='L506',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

def get_loader_3(mode='train', load_mode=0,
               saved_path=None, test_patient='L506',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_ed_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def get_loader_4(mode='train', load_mode=0,
               saved_path_1=None, saved_path_2=None,test_patient='L506',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_latent_dataset(mode, load_mode, saved_path_1,saved_path_2, test_patient, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


