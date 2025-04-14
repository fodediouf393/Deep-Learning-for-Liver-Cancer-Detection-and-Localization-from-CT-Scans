class Model(pl.LightningModule):
    def __init__(self, learning_rate=0.001, task="segmentation"):
        """
        Initialize the model.
        Args:
            learning_rate (float): Learning rate for the optimizer.
            task (str): Task type - 'segmentation' or 'classification'.
        """
        super().__init__()
        self.lr = learning_rate
        self.task = task
        self.net = UNet(in_channels=1, n_classes=3)
        # self.net = VGG16(num_classes=3)
        self.dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = Dice(num_classes=3, ignore_index=0)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=3) if self.task == "classification" else None
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3) if self.task == "classification" else None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)

    def compute_loss_and_metrics(self, output, target):
        """
        Compute loss and metrics based on the task.
        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.
        Returns:
            loss (torch.Tensor): Computed loss.
            metrics (dict): Computed metrics.
        """
        metrics = {}
        if self.task == "classification":
            loss = self.cross_entropy(output, target.squeeze(1))
            if self.train_accuracy:
                softmax = nn.Softmax(dim=1)
                probs = softmax(output)
                probs = torch.argmax(probs, dim=1).unsqueeze(1)
                self.train_accuracy(probs, target)
                metrics["accuracy"] = self.train_accuracy.compute()
        elif self.task == "segmentation":
            loss = self.dice_loss(output, target)
            softmax = nn.Softmax(dim=1)
            probs = softmax(output)
            probs = torch.argmax(probs, dim=1).unsqueeze(1)
            dice = self.dice(probs, target)
            metrics["dice_score"] = dice
        else:
            raise ValueError("Task must be either 'segmentation' or 'classification'.")
        return loss, metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.net(x)
        loss, metrics = self.compute_loss_and_metrics(output, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(f"train_{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.net(x)
        loss, metrics = self.compute_loss_and_metrics(output, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_epoch_end(self):
        if self.task == "classification":
            self.train_accuracy.reset()
            self.val_accuracy.reset()
