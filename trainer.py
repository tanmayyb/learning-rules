import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
epochs = 10

RULE_SELECT = 'hebb'
NUM_INPUTS = 784
NUM_HIDDEN = 100
NUM_OUTPUTS = 10
ACTIVATION_TYPE = 'sigmoid'
BIAS = False



# download/load train dataset
train_loader = DataLoader(
    datasets.MNIST(
        './data', train=True, download=True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),(0.3081,)
            )
        ])
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,    
)

# download/load train dataset
test_loader = DataLoader(
    datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),(0.3081,)
            )        
        ]),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)



if RULE_SELECT == 'hebb':

    # from models.FA import *

    # model = FANetwork(
    #     in_features=784, 
    #     num_layers=2, 
    #     num_hidden_list=[1000, 10]
    # ).to(device)

    from rules.Hebbian import HebbianNetwork

    model = HebbianNetwork(
        num_inputs=NUM_INPUTS,
        num_hidden=NUM_HIDDEN,
        num_outputs=NUM_OUTPUTS,
        clamp_output=False,
        bias=BIAS,
    ).to(device)

else:
    raise NotImplementedError("Selected Rule does not exist!")




# def train_model(model, train_loader.)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True,
)

loss_crossentropy = torch.nn.CrossEntropyLoss()

for epoch in tqdm(range(epochs)):
    for idx_batch, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(BATCH_SIZE, -1)

        # autograd vars
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs.to(device))

        loss = loss_crossentropy(outputs, targets.to(device))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # if (...):
        #     logging...




