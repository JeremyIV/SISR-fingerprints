from torchvision import transforms

# taken from:
# https://github.com/vishal3477/Reverse_Engineering_GMs/

# there's a fingerprint estimation network
# and there's an attribution network
############################################################################################
## Model architectures
############################################################################################


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    # print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return real, imag


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [
            nn.Sequential(
                nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        ]
        for i in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        residual_1 = residual.clone()

        residual_gray = (
            0.299 * residual_1[:, 0, :, :].clone()
            + 0.587 * residual_1[:, 1, :, :].clone()
            + 0.114 * residual_1[:, 2, :, :].clone()
        )

        thirdPart_fft_1 = torch.rfft(residual_gray, signal_ndim=2, onesided=False)

        thirdPart_fft_1_orig = thirdPart_fft_1.clone()

        thirdPart_fft_1[:, :, :, 0], thirdPart_fft_1[:, :, :, 1] = fftshift(
            thirdPart_fft_1[:, :, :, 0], thirdPart_fft_1[:, :, :, 1]
        )
        thirdPart_fft_1 = torch.sqrt(
            thirdPart_fft_1[:, :, :, 0] ** 2 + thirdPart_fft_1[:, :, :, 1] ** 2
        )
        n = 25
        (_, w, h) = thirdPart_fft_1.shape
        half_w, half_h = int(w / 2), int(h / 2)
        thirdPart_fft_2 = thirdPart_fft_1[
            :, half_w - n : half_w + n + 1, half_h - n : half_h + n + 1
        ].clone()
        thirdPart_fft_3 = thirdPart_fft_1.clone()
        thirdPart_fft_3[:, half_w - n : half_w + n + 1, half_h - n : half_h + n + 1] = 0
        max_value = torch.max(thirdPart_fft_3)
        thirdPart_fft_4 = thirdPart_fft_1.clone()
        thirdPart_fft_4 = torch.transpose(thirdPart_fft_4, 1, 2)
        return (
            thirdPart_fft_1,
            thirdPart_fft_2,
            max_value,
            thirdPart_fft_1_orig,
            residual,
            thirdPart_fft_4,
            residual_gray,
        )


class encoder(torch.nn.Module):
    def __init__(self, num_hidden, num_classes):
        super(encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(65536, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out)
        out1 = self.fc2(out)
        # Sigmoid seems dumb here, since we're training with softmax-crossEntropy
        # which already normalizes the values.
        # out1 = torch.sigmoid(out1)
        # return out1, out
        return out1

    def transfer_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        state_dict["fc2.weight"] = self.fc2.weight
        state_dict["fc2.bias"] = self.fc2.bias
        self.load_state_dict(state_dict)


####################################################################
## Training routine
####################################################################


preprocess = transforms.ToTensor()


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


def train(
    model, model_2, train_dataset, val_dataset, num_epochs, learning_rate, save_dir
):
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    l1 = torch.nn.MSELoss().to(device)
    l_c = torch.nn.CrossEntropyLoss().to(device)

    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

    def _train(batch, labels):
        model.train()
        model_2.train()
        y, low_freq_part, max_value, y_orig, residual, y_trans, residual_gray = model(
            batch.type(torch.cuda.FloatTensor)
        )

        y_2 = torch.unsqueeze(y.clone(), 1)
        classes = model_2(y_2)
        classes_f = torch.max(classes, dim=1)[0]

        n = 25
        zero = torch.zeros([y.shape[0], 2 * n + 1, 2 * n + 1], dtype=torch.float32).to(
            device
        )
        zero_1 = torch.zeros(residual_gray.shape, dtype=torch.float32).to(device)
        loss1 = 0.05 * l1(low_freq_part, zero).to(device)
        loss2 = -0.001 * max_value.to(device)
        loss3 = 0.01 * l1(residual_gray, zero_1).to(device)
        loss_c = 10 * l_c(classes, labels.type(torch.cuda.LongTensor))
        loss5 = 0.1 * l1(y, y_trans).to(device)

        loss = loss1 + loss2 + loss3 + loss_c + loss5
        optimizer.zero_grad()
        optimizer_2.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_2.step()
        return loss, classes

    def _test(batch, labels):
        model.eval()
        model_2.eval()
        with torch.no_grad():
            (
                y,
                low_freq_part,
                max_value,
                y_orig,
                residual,
                y_trans,
                residual_gray,
            ) = model(batch.type(torch.cuda.FloatTensor))
            y_2 = torch.unsqueeze(y.clone(), 1)
            classes = model_2(y_2)
            classes_f = torch.max(classes, dim=1)[0]

            n = 25
            zero = torch.zeros(
                [y.shape[0], 2 * n + 1, 2 * n + 1], dtype=torch.float32
            ).to(device)
            zero_1 = torch.zeros(residual_gray.shape, dtype=torch.float32).to(device)
            loss1 = 0.5 * l1(low_freq_part, zero).to(device)
            loss2 = -0.001 * max_value.to(device)
            loss3 = 0.01 * l1(residual_gray, zero_1).to(device)
            loss_c = 10 * l_c(classes, labels.type(torch.cuda.LongTensor))
            loss5 = 0.1 * l1(y, y_trans).to(device)
            loss = loss1 + loss2 + loss3 + loss_c + loss5
        return loss, classes

    train_dataset = preprocessed_dataset(train_dataset, preprocess)
    val_dataset = preprocessed_dataset(val_dataset, preprocess)
    batch_size = train_opt["batch_size"]
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    dataloaders = {"train": train_dataloader, "val": val_dataloader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    model = cnn.to(device)
    model_2 = cnn.to(device)

    since = time.time()

    best_acc = -1  # -1 handles edge case where best acc is zero

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            print(phase)

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    loss, class_logits = _train(inputs, labels)
                else:
                    loss, class_logits = _test(inputs, labels)
                preds = torch.max(class_logits, dim=1)[1].squeeze()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            save_dir.mkdir(parents=True, exist_ok=True)

            if phase == "train":
                model_path = save_dir / "fen_latest.pt"
                torch.save(model.state_dict(), model_path)
                model_2_path = save_dir / "encoder_latest.pt"
                torch.save(model_2.state_dict(), model_2_path)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = save_dir / "fen_best.pt"
                torch.save(model.state_dict(), model_path)
                model_2_path = save_dir / "encoder_best.pt"
                torch.save(model_2.state_dict(), model_2_path)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))


FEN_CLASSIFIER_EXPERIMENTS_PATH = Path("classification/classifiers/experiments/FEN")


def train_classifier(opt, train_dataset, val_dataset, save_dir):
    num_classes = train_dataset.ordered_labels
    model = DnCNN()
    model_2 = encoder(num_hidden=512, num_classes=num_classes)
    train_dataset.patch_size = 224  #
    val_dataset.patch_size = 224  # TODO: what patch size?
    if "pretrained_path" in opt:
        pretrained_path = Path(opt["pretrained_path"])
        model.load_state_dict(torch.load(pretrained_path / "fen_best.pt"))
        model_2.transfer_state_dict(torch.load(pretrained_path / "encoder_best.pt"))

    train_opt = opt["train"]
    num_epochs = train_opt["num_epochs"]
    learning_rate = train_opt["learning_rate"]

    train(
        model, model_2, train_dataset, val_dataset, num_epochs, learning_rate, save_dir
    )


##############################################################################################
## Saving/loading/bookeeping stuff
##############################################################################################


def save_classifier_to_database(
    classifier_name, train_dataset, val_dataset, save_dir, opt
):
    training_dataset_id = db.get_unique_row("dataset", {"name": train_dataset.name}).id
    val_dataset_id = db.get_unique_row("dataset", {"name": val_dataset.name}).id

    db.idempotent_insert_unique_row(
        "classifier",
        {
            "training_dataset_id": training_dataset_id,
            "validation_dataset_id": val_dataset_id,
            "name": classifier_name,
            "path": save_dir / "model_best.pt",
            "type": "FEN",
            "opt": opt,
        },
    )


@CLASSIFIER_REGISTRY.register()
class FEN:
    def __init__(self, classifier_name, ordered_labels, model, model_2):
        pass
        self.name = classifier_name
        self.ordered_labels = ordered_labels
        self.model = model
        self.model_2 = model_2

    def __call__(self, image):
        (
            y,
            low_freq_part,
            max_value,
            y_orig,
            residual,
            y_trans,
            residual_gray,
        ) = self.model(preprocess(image))
        y_2 = torch.unsqueeze(y.clone(), 1)
        class_logits = self.model_2(y_2)
        probabilities = self.softmax(class_logits).detach().cpu().numpy().squeeze()
        feature = y_2.detach().cpu().numpy().squeeze()
        return probabilities, feature

    @staticmethod
    def train_and_save_classifier(
        classifier_opt, train_dataset, val_dataset, mode="both"
    ):
        save_dir = FEN_CLASSIFIER_EXPERIMENTS_PATH / opt["name"]
        classifier_name = opt["name"]

        if mode != "test":
            train_classifier(opt, train_dataset, val_dataset, save_dir)
        if mode != "train":
            save_classifier_to_database(
                classifier_name, train_dataset, val_dataset, save_dir, opt
            )
        return classifier_name

    @staticmethod
    def load_classifier(classifier_row):
        training_dataset_row = db.get_unique_row(
            "dataset", {"id": classifier_row.training_dataset_id}
        )
        num_classes = len(training_dataset_row.ordered_labels)
        model = DnCNN()
        model.load_state_dict(torch.load(classifier_row.path / "fen_best.pt"))
        model_2 = encoder(num_hidden=512, num_classes=num_classes)
        model_2.load_state_dict(torch.load(classifier_row.path / "encoder_best.pt"))
        return FEN(
            classifier_row.name, training_dataset_row.ordered_labels, model, model_2
        )
