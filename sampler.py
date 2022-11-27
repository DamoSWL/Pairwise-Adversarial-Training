from torch.utils.data import Sampler
from collections import OrderedDict
import random



class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class TaskSampler(metaclass=Singleton):
    def __init__(self, unique_classes, args):
        self.unique_classes = sorted(unique_classes)
        self.args = args
        self.k_shot = args['k_shot']
        self.n_way = args['n_way']
        self.counter = 0
        self.sampled_classes = None
        self.sample_freq = 1 #self.get_sampling_frequency()

    def get_sampling_frequency(self):
        # because of the Singleton and two dataloaders: source and target
        # need to make sure sampled classes align between souce and target
        if self.args.self_train is True:
            freq = 2
        else:
            freq = 1
        return freq

    def sample_N_classes_as_a_task(self):
        # mod-2 because both the source and target domains are using the same sampler
        # sometimes need to make sure they sample the same set of classes
        if self.counter % self.sample_freq == 0:
            self.sampled_classes = random.sample(self.unique_classes, self.n_way)

        self.counter += 1
        return self.sampled_classes


class N_Way_K_Shot_BatchSampler(Sampler):
    def __init__(self, y, max_iter, task_sampler):
        self.y = y
        self.max_iter = max_iter
        self.task_sampler = task_sampler
        self.label_dict = self.build_label_dict()
        self.unique_classes_from_y = sorted(set(self.y))

    def build_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y):
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)
        return label_dict

    def sample_examples_by_class(self, cls):
        if cls not in self.label_dict:
            return []

        if self.task_sampler.k_shot <= len(self.label_dict[cls]):
            sampled_examples = random.sample(self.label_dict[cls],
                                             self.task_sampler.k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(self.label_dict[cls],
                                              k=self.task_sampler.k_shot)  # sample with replacement
        return sampled_examples

    def __iter__(self):
        for _ in range(self.max_iter):
            batch = []
            classes = self.task_sampler.sample_N_classes_as_a_task()
            if len(batch) == 0:
                for cls in classes:
                    samples_for_this_class = self.sample_examples_by_class(cls)
                    batch.extend(samples_for_this_class)

            yield batch

    def __len__(self):
        return self.max_iter

