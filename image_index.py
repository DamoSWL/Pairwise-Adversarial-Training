from os import path


def get_image_files(images_file_path, datasets_dir):
    image_files = get_image_filenames_and_labels(images_file_path, datasets_dir)
    return image_files


def get_image_filenames_and_labels(images_file_path, datasets_dir):
    image_files = open(images_file_path).readlines()
    image_files = [path.join(datasets_dir, img_file) for img_file in image_files]
    image_files = list(map(parse_filname_and_label, image_files))
    return image_files


def parse_filname_and_label(img_index):
    img_path, label_id = img_index.split()
    label_id = int(label_id)
    return img_path, label_id

