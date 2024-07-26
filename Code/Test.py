import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import KFold
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session


# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define a simple model for demonstration purposes
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the train_k_fold function
def train_k_fold(config, dataset, model_name, model_folder_path, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(config["batch_size"]), sampler=train_subsampler, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(config["batch_size"]), sampler=val_subsampler, num_workers=4, pin_memory=True)

        model = SimpleModel(input_size=10, hidden_size=config["hidden_size"], output_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        for epoch in range(10):  # Simplified training loop for demonstration
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    mean_val_loss = np.mean(val_losses)
    session.report({"mean_val_loss": mean_val_loss})

# Define the dataset (random data for demonstration)
dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))

# Define hyperparameter search space
config = {
    "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([32, 64, 128]),
}

scheduler = ASHAScheduler(
    metric="mean_val_loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

# Configure the tuner
tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_k_fold, dataset=dataset, model_name="SimpleModel", model_folder_path="./"),
        resources={"cpu": 4, "gpu": 1}
    ),
    param_space=config,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=10,
        max_concurrent_trials=1  # Limit to 1 concurrent trial
    ),
    run_config=ray.air.config.RunConfig(
        name="tune_k_fold",
        verbose=1
    )
)

# Run the tuner
results = tuner.fit()

# Get the best result
best_result = results.get_best_result()

# Get the best configuration and the best metrics
best_config = best_result.config
best_metrics = best_result.metrics

# Print and save the best configuration and best validation loss
print("Best hyperparameters found were: ", best_config)
print("Best validation loss found was: ", best_metrics['mean_val_loss'])

with open(f'./best_config.txt', 'w') as f:
    f.write(f"Best hyperparameters: {best_config}\n")
    f.write(f"Best validation loss: {best_metrics['mean_val_loss']}\n")