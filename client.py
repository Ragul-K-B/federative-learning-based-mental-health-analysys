import flwr as fl
import torch
import numpy as np
from model import Model
from dataloader import load_data

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, file_path, input_dim, num_classes):
        self.model = Model(input_dim=input_dim, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader, self.test_loader = load_data(file_path, input_dim, num_classes)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
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

    def predict(self, user_input):
        self.model.eval()
        user_input = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(user_input)
        return output.numpy()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Client for Heart Disease Prediction.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimension of the input features.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')

    args = parser.parse_args()

    file_path = args.file_path
    input_dim = args.input_dim
    num_classes = args.num_classes

    client = HeartDiseaseClient(file_path, input_dim, num_classes)

    # Ensure user input has the correct number of features
    user_input = np.array([
        float(input("ENTER SLEEP: ")),
        float(input("ENTER APPETITE: ")),
        float(input("ENTER INTEREST: ")),
        float(input("ENTER FATIGUE: ")),
        float(input("ENTER WORTHLESS: ")),
        float(input("ENTER CONCENTRATION: ")),
        float(input("ENTER AGITATION: ")),
        float(input("ENTER SUICIDE_THOUGHTS: ")),
        float(input("ENTER SLEEP_DISTURBANCE: ")),
        float(input("ENTER AGRESSION: ")),
        float(input("ENTER PANIC ATTACK: ")),
        float(input("ENTER HOPELESSNESS: ")),
        float(input("ENTER RESTLESSNESS: ")),
        float(input("ENTER LOW ENERGY ")),

    ])


    if len(user_input) != input_dim:
        raise ValueError(f"Expected {input_dim} features, but got {len(user_input)}")

    prediction = client.predict(user_input)
    print(f"Prediction: {prediction}")
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest value

    # Use the index to determine the corresponding class
    if predicted_class_index == 0:
        print("no")
    elif predicted_class_index == 1:
        print("mild")
    elif predicted_class_index == 2:
        print("moderate")
    else:
        print("severe")
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
