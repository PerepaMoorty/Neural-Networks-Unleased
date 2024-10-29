import torch

# Getting Which Device to Use
device = torch.device('cuda') if torch.cuda else torch.device('cpu')

def Sobel_Filter(image):
    # Converting the Image to an PyTorch Tensor
    image_array = torch.tensor(image, dtype=torch.float32).to(device)
    
    # Horizontal Sobel Filter
    sobel_x = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).to(device)

    # Vertical Sobel Filter
    sobel_y = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).to(device)
    
    # Getting the Image Height and Width
    image_height =  image_array.shape[1]
    image_width =  image_array.shape[2]