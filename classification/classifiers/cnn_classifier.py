# cnn_classifier.py
from classification.utils.registry import CLASSIFIER_REGISTRY
from torch import nn
from torch.utils.data import Dataset, DataLoader
from classification.classifiers import arch
from pathlib import Path
import torch.optim as optim
import time
import torch
import tqdm
import database.api as db


class preprocessed_dataset(Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, metadata = self.dataset[index]
        label_index = self.dataset.ordered_labels.index(label)
        return self.preprocess(image), label_index


device = torch.device("cuda")

have_printed = False


def train(cnn, train_dataset, val_dataset, num_epochs, train_opt, save_dir):
    train_dataset = preprocessed_dataset(train_dataset, cnn.preprocess)
    val_dataset = preprocessed_dataset(val_dataset, cnn.preprocess)
    batch_size = train_opt["batch_size"]
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    # create optimizer
    learning_rate_start = train_opt["learning_rate_start"]
    learning_rate_end = train_opt["learning_rate_end"]
    optimizer = optim.Adamax(cnn.parameters(), lr=learning_rate_start)
    # create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    # for phase in train, val:
    cnn = cnn.to(device)

    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        if optimizer.param_groups[0]["lr"] < learning_rate_end:
            print("Model appears to have stopped improving.")
            print("Ending training early.")
            break

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                cnn.train()  # Set cnn to training mode
            else:
                cnn.eval()  # Set cnn to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            print(phase)

            # TODO: understand why this gets 3x slower after first epoch
            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                criterion = nn.CrossEntropyLoss()
                # TODO: why is this 3x slower in the second epoch?
                inputs = inputs.to(device)
                labels = labels.to(device)

                global have_printed
                if not have_printed:
                    print("Input format::")
                    print(inputs.shape)
                    print(inputs.min(), inputs.max())
                    print(labels)
                    have_printed = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    probabilities, features = cnn(inputs)
                    _, preds = torch.max(probabilities, 1)
                    loss = criterion(probabilities, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                scheduler.step(epoch_loss)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            save_dir.mkdir(parents=True, exist_ok=True)

            if phase == "train":
                cnn_path = save_dir / "model_latest.pt"
                torch.save(cnn.state_dict(), cnn_path)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                cnn_path = save_dir / "model_best.pt"
                torch.save(cnn.state_dict(), cnn_path)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))


CNN_CLASSIFIER_EXPERIMENTS_PATH = Path("classification/classifiers/experiments/CNN")


@CLASSIFIER_REGISTRY.register()
class CNN:
    def __init__(self, name, ordered_labels, cnn):
        self.name = name
        self.ordered_labels = ordered_labels
        self.cnn = cnn

    def __call__(self, image):
        img = self.cnn.preprocess(image).unsqueeze(0)
        probabilities, feature = self.cnn(img)
        probabilities = probabilities.detach().cpu().numpy().squeeze()
        feature = feature.detach().cpu().numpy().squeeze()
        return probabilities, feature

    @staticmethod
    def train_and_save_classifier(opt, train_dataset, val_dataset):
        cnn_opt = opt["cnn"].copy()
        cnn_opt["num_classes"] = len(train_dataset.ordered_labels)
        cnn = arch.get_cnn(cnn_opt)
        if "pretrained_path" in opt:
            state_dict = torch.load(opt["pretrained_path"])
            cnn.transfer_state_dict(state_dict)

        train_opt = opt["train"]
        save_dir = CNN_CLASSIFIER_EXPERIMENTS_PATH / opt["name"]
        num_pretrain_epochs = train_opt["num_pretrain_epochs"]
        num_full_train_epochs = train_opt["num_full_train_epochs"]

        cnn.freeze_all_but_last_layer()
        train(
            cnn=cnn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_pretrain_epochs,
            train_opt=train_opt,
            save_dir=save_dir,
        )
        cnn.unfreeze_all()
        train(
            cnn=cnn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_full_train_epochs,
            train_opt=train_opt,
            save_dir=save_dir,
        )
        # TODO: write the classifier to the database
        classifier_name = opt["name"]
        training_dataset_id = db.get_unique_row(
            "dataset", {"name": train_dataset.name}
        ).id
        val_dataset_id = db.get_unique_row("dataset", {"name": val_dataset.name}).id

        db.idempotent_insert_unique_row(
            "classifier",
            {
                "training_dataset_id": training_dataset_id,
                "validation_dataset_id": val_dataset_id,
                "name": classifier_name,
                "path": save_dir / "model_best.pt",
                "type": "CNN",
                "opt": opt,
            },
        )
        return classifier_name

    @staticmethod
    def load_classifier(classifier_row):
        # TODO: get num_classes from the dataset
        training_dataset_row = db.get_unique_row(
            "dataset", {"id": classifier_row.training_dataset_id}
        )
        num_classes = len(training_dataset_row.ordered_labels)
        cnn_opt = classifier_row.opt["cnn"].copy()
        cnn_opt["num_classes"] = num_classes
        cnn = arch.get_cnn(cnn_opt)
        state_dict_path = classifier_row.path
        state_dict = torch.load(state_dict_path)
        cnn.load_state_dict(state_dict)
        return CNN(classifier_row.name, training_dataset_row.ordered_labels, cnn)
