train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,

    url='https://storage.googleapis.com/cvdf-datasets/mnist/'
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform,
    url='https://storage.googleapis.com/cvdf-datasets/mnist/'
)
