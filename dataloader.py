from data_list import ImageList
from sampler import N_Way_K_Shot_BatchSampler, TaskSampler
import image_index
import torch.utils.data as util_data
from torchvision import transforms



class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))



def GetDataLoader(cfg,
                images_file_path, 
                batch_size, 
                dataset_dir,
                is_train=True, 
                is_source=True,
                sample_mode_with_ground_truth_labels=False, 
                drop_flag=False):

    image_files = image_index.get_image_files(images_file_path, dataset_dir)
    data_sampler = None

    resize_size = 256
    crop_size = 224

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train is not True: 
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])

        images = ImageList(image_files, transform=transformer)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False,
                                             num_workers=2)
    else: 
        crop_options = {
            'RandomResizedCrop': transforms.RandomResizedCrop,
            'RandomCrop': transforms.RandomCrop
        }
        crop_type = crop_options['RandomResizedCrop']
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          crop_type(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

        images = ImageList(image_files, transform=transformer) 


        if sample_mode_with_ground_truth_labels is False:
            images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=2,drop_last=drop_flag)
        elif sample_mode_with_ground_truth_labels is True:
            images_loader, data_sampler = nway_kshot_dataloader(images, cfg)
            if is_source is False:
                print('warning! you are sampling with ground-truth labels')
        else:
            raise ValueError('could not create dataloader under the given config')

    return images_loader, data_sampler



def nway_kshot_dataloader(images, args):
    task_sampler = TaskSampler(set(images.labels), args)
    n_way_k_shot_sampler = N_Way_K_Shot_BatchSampler(images.labels, args['max_iter'], task_sampler)
    meta_loader = util_data.DataLoader(images, shuffle=False, batch_sampler=n_way_k_shot_sampler,num_workers=2)
    return meta_loader, n_way_k_shot_sampler