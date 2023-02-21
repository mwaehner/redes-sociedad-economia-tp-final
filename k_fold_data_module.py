from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset

from students_dataset import StudentsDataset


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class KFoldDataModule(BaseKFoldDataModule):
    def __init__(self,
                 dataset: StudentsDataset ,
                 batch_size: int = 1,
                 seed: int = 42,
                 num_workers: int = 2,
                 number_of_folds: int = 5
                 ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.splits = self.setup_folds(number_of_folds)

        self.train_fold: Dataset  # Holds current training fold
        self.val_fold: Dataset  # Holds current validation fold

    def setup_folds(self, number_of_folds: int) -> list:
        return [
            split for split in KFold(number_of_folds, shuffle=True, random_state=self.seed).split(
                range(len(self.dataset))
            )
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]

        self.dataset.train_indices = train_indices
        self.train_fold = Subset(self.dataset, train_indices)
        self.val_fold = Subset(self.dataset, val_indices)
        print(f"Val fold size indices are {val_indices[:5]}...")

    def train_dataloader(self) -> DataLoader:
        print(f"Train fold size is {len(self.train_fold)}")
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        print(f"Val fold size is {len(self.val_fold)}")
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_example_input(self):
        example = self.dataset[0]['image']
        return example.view(1, *list(example.shape))