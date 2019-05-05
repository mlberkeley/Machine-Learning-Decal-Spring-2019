# 1: Preprocessing

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 2: Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1)

    def forward(self, x):
        batch = list(x.size())[0]
        x = x.view(batch, -1)
        x = F.sigmoid(self.fc(x))
        return x

# 3: Postprocess

# 4: Written explanation