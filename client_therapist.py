import flwr as fl
import torch
import pandas as pd
import numpy as np
from model import Model
from dataloader import load_data

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, file_path, input_dim, num_classes, is_testing=False):
        self.model = Model(input_dim=input_dim, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.is_testing = is_testing
        self.file_path = file_path

        if is_testing:

            self.test_loader = self._load_test_data(file_path, input_dim)
        else:

            self.train_loader, self.test_loader = load_data(file_path, input_dim, num_classes)

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Training function"""
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        """Evaluation function"""
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}

    def predict(self, data):
        """Prediction function"""
        self.model.eval()
        data = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(data)
        return outputs.numpy()

    def _load_test_data(self, file_path, input_dim):
        """Load test data for prediction"""
        df = pd.read_csv(file_path)
        self.original_df = df
        X_test = df.iloc[:, :input_dim].values
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=1,
            shuffle=False
        )

    def save_predictions(self, predictions):
        """Save predictions to file"""
        predicted_indices = np.argmax(predictions, axis=1)
        self.original_df['Predicted Class Index'] = predicted_indices
        self.original_df.to_csv(self.file_path, index=False)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Client for Heart Disease Prediction.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimension of the input features.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    parser.add_argument('--is_testing', action='store_true', help='Indicates whether the client is running in testing mode.')

    args = parser.parse_args()

    file_path = args.file_path
    input_dim = args.input_dim
    num_classes = args.num_classes
    is_testing = args.is_testing

    client = HeartDiseaseClient(file_path, input_dim, num_classes, is_testing=is_testing)

    if is_testing:

        predictions = []
        for data_batch in client.test_loader:
            data = data_batch[0]
            prediction = client.predict(data)
            predictions.extend(prediction)
        client.save_predictions(np.array(predictions))
        print("Predictions saved in the original file.")
    else:

        user_input = np.array([float(input(f"ENTER {feature}: ")) for feature in [
            "SLEEP", "APPETITE", "INTEREST", "FATIGUE", "WORTHLESS",
            "CONCENTRATION", "AGITATION", "SUICIDE_THOUGHTS", "SLEEP_DISTURBANCE",
            "AGRESSION", "PANIC ATTACK", "HOPELESSNESS", "RESTLESSNESS", "LOW ENERGY"]])

        if len(user_input) != input_dim:
            raise ValueError(f"Expected {input_dim} features, but got {len(user_input)}")


        prediction = client.predict(user_input[np.newaxis, :])
        predicted_class_index = np.argmax(prediction)

        # Map predicted index to the corresponding class
        class_labels = ["no", "mild", "moderate", "severe"]
        print(f"Prediction: {class_labels[predicted_class_index]}")

        fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
