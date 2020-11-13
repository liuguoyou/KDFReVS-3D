import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as T





def get_video_data_loaders_gru(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 64)
    crop = cfgs.get('crop', None)
    max_num_views = cfgs.get('max_num_views', 30)
    min_num_views = cfgs.get('min_num_views', 15)

    num_views = np.random.randint(min_num_views, max_num_views)

    print("random view number : {}".format(num_views))

    run_train = cfgs.get('run_train', False)
    train_data_dir = cfgs.get('train_data_dir', './models')
    val_data_dir = cfgs.get('val_data_dir')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './models/test')

    load_gt_depth = cfgs.get('load_gt_depth', False)
    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    load_type = cfgs.get('load_type', 0)

    train_loader = val_loader = test_loader = None
    if load_gt_depth:
        pass
        # get_loader = lambda **kargs: get_paired_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    else:
        get_loader = lambda **kargs: get_video_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop,
                                                      view_num=num_views,load_type=load_type)

    if run_train:
        # print(os.getcwd())
        assert os.path.isdir(train_data_dir), "Training models directory does not exist: %s" % train_data_dir
        assert os.path.isdir(val_data_dir), "Validation models directory does not exist: %s" % val_data_dir
        print(f"Loading training models from {train_data_dir}")
        train_loader = get_loader(data_dir=train_data_dir, is_validation=False)
        print(f"Loading validation models from {val_data_dir}")
        val_loader = get_loader(data_dir=val_data_dir, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing models directory does not exist: %s" % test_data_dir
        print(f"Loading testing models from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)

    return train_loader, val_loader, test_loader


def multi_view_collate(batch):
    "Prepare batch with 1 image and n_poses masks and poses per item"
    bs = len(batch)
    n_poses = batch[0][0].size(0)

    idxs = torch.randint(0, n_poses, size=(bs,))
    imgs, poses, masks = zip(*[(img[i], view, mask) for (img, view, mask), i in zip(batch, idxs)])

    imgs = torch.stack(imgs)
    poses = torch.cat(poses, dim=0)
    masks = torch.cat(masks, dim=0)

    return imgs, poses, masks


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def make_dataset_youtube_and_300vw(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    videos = []

    dir1 = os.path.join(dir, "300VW_annot_224")
    dir2 = os.path.join(dir, "youtube_annot_224")

    for ddir in [dir1, dir2]:
        for sf in sorted(os.listdir(ddir)):
            single_video_frames = []
            for fname in sorted(os.listdir(os.path.join(ddir, sf, "crop"))):
                if is_image_file(fname):
                    fpath = os.path.join(ddir, sf, "crop", fname)
                    single_video_frames.append(fpath)
            videos.append((sf, single_video_frames))
    return videos

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    videos = []
    for sf in sorted(os.listdir(dir)):
        single_video_frames = []
        for fname in sorted(os.listdir(os.path.join(dir, sf))):
            if is_image_file(fname) and fname.startswith("im_"):
                fpath = os.path.join(dir, sf, fname)
                single_video_frames.append(fpath)
        videos.append((sf, single_video_frames))
    return videos


def make_dataset_makeall(dir, view_num, load_type=0):
    videos = make_dataset_youtube_and_300vw(dir)
    multi_view_sets = []

    for vname, frames in videos:
        if len(frames) < view_num:
            continue
            #frames = padding_frames(frames, view_num)

        if load_type == 1:
            random.shuffle(frames)
        sets_num = int(len(frames) / view_num)
        for i in range(sets_num):
            multi_view_sets.append(frames[i * view_num:(i + 1) * view_num])

    return multi_view_sets



def padding_frames(frames, needs_len):
    if len(frames) == 1:
        return frames * needs_len
    else:
        padding_num = int(needs_len / (len(frames)-1))
    padding_frames = []
    for i in range(padding_num + 1):
        if i % 2 == 0:
            padding_frames += frames
        else:
            padding_frames += frames[1:-1][::-1]
    return padding_frames

def make_dataset_makepart(dir, view_num, load_type=0):
    videos = make_dataset(dir)
    multi_view_sets = []

    for vname, frames in videos:
        if len(frames) < view_num:
            frames = padding_frames(frames, view_num)

        #start = np.random.randint(0, len(frames) - view_num)

        start = 0
        multi_view_sets.append(frames[start: start+view_num])

    return multi_view_sets

def make_dataset_makepart_balance(dir, view_num, min_views = 60, load_type=0):
    assert view_num < min_views
    videos = make_dataset_youtube_and_300vw(dir)
    multi_view_sets = []


    print("found {} videos".format(len(videos)))
    for vname, frames in videos:
        if len(frames) < min_views:
            frames = padding_frames(frames, min_views)

        if load_type == 1:
            random.shuffle(frames)
        start = np.random.randint(0, len(frames) - view_num)

        multi_view_sets.append(frames[start: start+view_num])

    return multi_view_sets

def make_dataset_makepart_balance_sample(dir, view_num, min_views = 150, sample_step=8, load_type=0):
    assert view_num < min_views
    videos = make_dataset_youtube_and_300vw(dir)
    multi_view_sets = []


    print("found {} videos".format(len(videos)))
    for vname, frames in videos:
        if len(frames) < min_views:
            frames = padding_frames(frames, min_views)


        start = np.random.randint(0, len(frames) - view_num * sample_step - 5)

        multi_view_sets.append(frames[start: start + view_num*sample_step : sample_step])

    return multi_view_sets

def make_dataset_validate(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
    return images


class VideoDataSets(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, view_nums=30, load_type=0):
        super(VideoDataSets, self).__init__()
        self.root = data_dir
        self.is_validation = is_validation
        self.view_nums = view_nums
        self.load_type = load_type  # 0 连续帧 1 随机采样
        print("current load type : {}".format(self.load_type))
        #self.paths = make_dataset_makeall(data_dir, self.view_nums, load_type=self.load_type) if not self.is_validation else make_dataset_validate(data_dir)
        self.paths = make_dataset_makepart_balance(data_dir, self.view_nums, load_type=self.load_type) if not self.is_validation else make_dataset_validate(data_dir)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop


    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(
                    self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def load1(self, index):
        fpath = self.paths[0:15]
        images = []
        hflip = not self.is_validation and np.random.rand() > 0.5
        for i in range(0, 15):
            o = Image.open(fpath[i])
            images.append(self.transform(o, hflip))

        images = torch.stack(images)
        return images
    def load2(self, index):
        fpath = self.paths[index % self.size]
        img = Image.open(fpath).convert('RGB')
        hflip = not self.is_validation and np.random.rand() > 0.5
        return self.transform(img, hflip=hflip)

    def __getitem__(self, index):
        if self.is_validation:
            return self.load2(index)
        else:
            video_frame_paths = self.paths[index]

            #print(video_frame_paths)
            images = []

            hflip = not self.is_validation and np.random.rand() > 0.5

            for i in range(0, self.view_nums):
                o = Image.open(video_frame_paths[i])
                images.append(self.transform(o, hflip))

            images = torch.stack(images)

            return images

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'



def get_video_loader(data_dir, view_num=30, is_validation=False,
                     batch_size=256, num_workers=1, image_size=256, crop=None,load_type=0):
    dataset = VideoDataSets(data_dir, image_size=image_size, crop=crop, view_nums=view_num, is_validation=is_validation,load_type=load_type)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        pin_memory=True
    )
    return loader
