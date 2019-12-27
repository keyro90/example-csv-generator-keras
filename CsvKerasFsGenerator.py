import logging
import os
import uuid

import pandas as pd
from keras.utils import Sequence


class CsvKerasFsGenerator(Sequence):

    @staticmethod
    def __constrain(v: int, min_v: int = 1, max_v: int = 1):
        if min_v > max_v:
            raise ValueError(f'min_v {min_v} is greater than max_v {max_v}')
        if v < min_v:
            return min_v
        if v > max_v:
            return max_v
        return v

    def __init__(self, path_filename: str, path_splitted_files: str, columns: list, x_exclude_cols: list, y_column: str,
                 batch_size: int = 1, sep: str = ','):
        logging.debug(f'Checking dir and file : {path_splitted_files},{path_filename}')
        if not os.path.isdir(path_splitted_files):
            raise ValueError(f'path_splitted_files = {path_splitted_files} is not a valid dir or doesn\'t exist.')
        if not os.path.isfile(path_filename):
            raise ValueError(f'filename = {path_filename} is not a valid file or doesn\'t exist.')
        self.x_exclude_cols = x_exclude_cols
        self.y_column = y_column
        self.sep = sep
        self.filename = path_filename
        self.path_splitted_files = path_splitted_files
        self.is_init = False
        self.skip_head = False
        # starting count lines and then checking if batch_size is a divisor.
        count_lines = 0
        file_r = open(path_filename)
        line = file_r.readline()
        while line:
            line = file_r.readline()
            count_lines += 1
        file_r.close()
        logging.debug(f'Number lines file : {count_lines}.')
        self.columns = columns
        if count_lines <= 0:
            raise ValueError(f'No record found into {path_filename}')
        self.batch_size = self.__constrain(batch_size, 1, count_lines)
        if self.batch_size > count_lines:
            raise ValueError(f'Batch size {self.batch_size} is greater than {count_lines}')
        if count_lines % self.batch_size > 0:
            raise ValueError(f'Batch size {self.batch_size} is invalid because is not a divisor of {count_lines}')
        self.count_record = count_lines
        self.filename_uuid = uuid.uuid1()
        self.n_batches = self.count_record / self.batch_size
        self.means = {k: .0 for k in self.columns}
        self.std_devs = {k: .0 for k in self.columns}

    def deploy(self):
        # split file and encode string
        self.__split_file_and_encode()
        print(self.means)
        print(self.std_devs)
        self.is_init = True

    def __getitem__(self, index):
        if not self.is_init:
            raise RuntimeError('Training cannot be done. Have you forgot .deploy() on this generator? :)')
        dataset = pd.read_csv(self.__name_splitted(index), names=self.columns)
        for i in range(len(dataset)):
            for c in self.columns:
                if c not in self.x_exclude_cols and c != self.y_column:
                    dataset.loc[i, c] = (dataset.loc[i, c] - self.means[c]) / self.std_devs[c]
        xs = dataset.drop(self.x_exclude_cols + [self.y_column], axis=1).values
        ys = dataset[self.y_column].values
        return xs, ys

    def __len__(self):
        return int(self.n_batches)

    ### Private Methods ###

    # Scan all file, split into n parts, calculate means and standard deviations for each feature.
    def __split_file_and_encode(self):
        counter_label = 0
        partial_count = 0
        file_counter = 0
        file_pointer = open(self.__name_splitted(file_counter), 'w')
        total_count = 0
        dict_label = {}
        # chunksize=1 it's a bullshit because in anycase you need to explicit row number to access into cell value.
        # But I can use column name and I don't need to strip something.
        for df in pd.read_csv(self.filename, sep=self.sep, header=None, chunksize=1, names=self.columns):
            # every batch I'm going to create a csv file.
            if partial_count >= self.batch_size:
                partial_count = 0
                file_counter += 1
                file_pointer.close()
                file_pointer = open(self.__name_splitted(file_counter), 'w')
            line = []
            for c in self.columns:
                val = df.at[total_count, c]
                real_val = val
                # if I've got a string, I'll encode it into a number.
                if isinstance(val, str):
                    if val not in dict_label.keys():
                        dict_label[val] = counter_label
                        counter_label += 1
                    line.append(str(dict_label[val]))
                    real_val = dict_label[val]
                else:
                    line.append(str(val))
                self.means[c] += float(real_val) # actually it's only sum of values.
                self.std_devs[c] += float(real_val ** 2) # actually it's only sum of squares.
            line_str = self.sep.join(line) + '\r\n'
            file_pointer.write(line_str)
            partial_count += 1
            total_count += 1
        file_pointer.close()
        # means and standard deviations can be used while model is training.
        self.means = {k: v / total_count for k, v in self.means.items()}
        self.std_devs = {k: (v / total_count - self.means[k] ** 2) ** 0.5 for k, v in self.std_devs.items()}

    def __name_splitted(self, file_counter):
        return f'{self.path_splitted_files}/{self.filename_uuid}_{file_counter}.csv'

    def n_classes(self):
        return len(self.columns) - len(self.x_exclude_cols) - 1
