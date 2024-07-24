import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
from datetime import datetime, timezone

# Get the current date and time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
epochs = 2
rule_select = 'hebb'
num_inputs = 784
num_hidden = 100
num_outputs = 10
activation_type = 'relu'
bias = False
lr=1e-4
momentum=0.9
weight_decay=0.001
nesterov=True

import wandb
from wandb_config import USERNAME

configs = dict(
    rule_select = rule_select,
    num_inputs = num_inputs,
    num_hidden = num_hidden,
    num_outputs = num_outputs,
    activation_type = activation_type,
    bias = bias,
    lr = lr,
    momentum = momentum,
    weight_decay = weight_decay,
    nesterov = nesterov,
)

experiment_name = "hebb-test"

now = datetime.now(timezone.utc)
formatted_datetime = now.strftime("%y%m%d-%H%M")
run = wandb.init(
    entity=USERNAME,
    project="learning-rules",
    name=experiment_name+"-"+formatted_datetime,
    config=configs,
    resume="never",
)




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



if rule_select == 'hebb':
    from rules.Hebbian import HebbianNetwork

    model = HebbianNetwork(
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        clamp_output=False,
        bias=bias,
    ).to(device)

elif rule_select == 'fa':
    from rules.FA import FeedbackAlignmentPerceptron

    model = FeedbackAlignmentPerceptron(
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        bias=bias,
        activation_type=activation_type,
    ).to(device)

# elif rule_select == ''

else:
    raise NotImplementedError("Selected Rule does not exist!")




# def train_model(model, train_loader.)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr, momentum=momentum, 
    weight_decay=weight_decay, 
    nesterov=nesterov,
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

        run.log({"loss": loss.item(), "epoch": epoch})


# Save the model
torch.save(model.state_dict(), "models/model.pth")
wandb.save("models/model.pth")

# run.log_artifact("models/model.pth", name = "state_dict", type = "dict" )

run.finish()




