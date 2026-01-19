from torchvision import datasets, transforms

def test_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ensure 1 channel
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
