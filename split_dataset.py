import os
import random
import shutil


class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=0.6, test_ratio=0.2, show_progress=False):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = saved_dataset_dir + "/train/"
        self.saved_valid_dir = saved_dataset_dir + "/valid/"
        self.saved_test_dir = saved_dataset_dir + "/test/"


        self.train_ratio = train_ratio
        self.test_radio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        self.show_progress = show_progress

        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)

    def __get_label_names(self):
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def __get_all_file_path(self):
        all_file_path = []
        for file_type in self.__get_label_names():
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def __copy_files(self, dataset_type):
        type_path = ""
        type_saved_dir = ""
        if dataset_type == "train":
            type_path = self.__get_train_file_path()
            type_saved_dir = self.saved_train_dir
        elif dataset_type == "test":
            type_path = self.__get_test_file_path()
            type_saved_dir = self.saved_test_dir
        elif dataset_type == "valid":
            type_path = self.__get_valid_file_path()
            type_saved_dir = self.saved_valid_dir
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:
                shutil.copy(src_path, dst_path)
                if self.show_progress:
                    print("Copying file "+src_path+" to "+dst_path)

    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)

            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_radio)

            self.train_file_path.append([index, file_path_list[: train_num]])
            self.test_file_path.append([index, file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([index, file_path_list[train_num + test_num:]])

    def __get_train_file_path(self):
        self.__split_dataset()
        return self.train_file_path

    def __get_test_file_path(self):
        self.__split_dataset()
        return self.test_file_path

    def __get_valid_file_path(self):
        self.__split_dataset()
        return self.valid_file_path

    def start_splitting(self):
        self.__copy_files(dataset_type="train")
        self.__copy_files(dataset_type="test")
        self.__copy_files(dataset_type="valid")


if __name__ == '__main__':
    split_dataset = SplitDataset(dataset_dir="original_dataset",
                                 saved_dataset_dir="dataset",
                                 show_progress=True)
    split_dataset.start_splitting()