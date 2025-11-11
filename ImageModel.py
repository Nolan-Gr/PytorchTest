import torch
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.datasets import fetch_openml

class ImageModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=784, out_features=64) #mon X c'est un array de 784 et il fait 10 guess en sortie
        self.layer_2 = nn.Linear(in_features=64, out_features=10) #parmis les 10 guess il en choisi un

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

class Model:

    device = "cuda"

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target #j'ai ma data de 70000 images de nombres écrits à la main
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80%/20% normal

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device) #data into tensor parce que sinon Linear() comprend pas
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    model_0 = ImageModelV1().to(device)

    loss_fn = nn.MSELoss() #loss function avec l'absolute error MSE
    optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

    def accu(self, y_true, y_pred):
        correct = (y_true == y_pred).sum()
        acc = correct / len(y_pred) * 100
        return acc

    torch.manual_seed(34)
    cycles = 100

    for cycle in range(cycles):
        model_0.train()

        y_logits = model_0(X_train)
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accu(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_0.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_0(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accu(y_test,test_pred)

        if cycle % 10 == 0:
            print(
                f"Epoch: {cycle} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")