def train_esrgan(generator,
                 discriminator,
                 dataloader,
                 optimizer_G, optimizer_D,
                 adversarial_loss_weight=0.01, content_loss_weight=0.03,
                 num_epochs=100,
                 gradient_clip_value=0.5):

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    for epoch in range(num_epochs):
        for images, _ in dataloader:

            # training the discriminator
            real_output = discriminator(images)
            fake_output = discriminator(generator(images.detach()))
            disc_real_loss = adversarial_loss(real_output, torch.ones_like(real_output))
            disc_fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
            disc_loss = disc_real_loss + disc_fake_loss
            optimizer_D.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_value_(discriminator.parameters(), gradient_clip_value)
            optimizer_D.step()

            # training the generator
            fake_output = discriminator(generator(images))
            gen_adversarial_loss = adversarial_loss(fake_output, torch.ones_like(fake_output))
            gen_content_loss = content_loss(generator(images), images)
            gen_loss = adversarial_loss_weight * gen_adversarial_loss + content_loss_weight * gen_content_loss
            optimizer_G.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_value_(generator.parameters(), gradient_clip_value)
            optimizer_G.step()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def custom_loader(path):
    img = Image.open(path)
    return transform(img)

dataset = datasets.ImageFolder('/content/dataset', loader=custom_loader)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = ESRGANGenerator(num_rrdb=28, residual_scaling=0.15, init_variance=0.03).to(device)
discriminator = ESRGANDiscriminator(num_conv_layers=14).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Experiment 1
# exp1_generator = ESRGANGenerator(num_rrdb=8, residual_scaling=0.1, init_variance=0.01).to(device)
# exp1_discriminator = ESRGANDiscriminator(num_conv_layers=5).to(device)
# exp1_optimizer_G = optim.Adam(exp1_generator.parameters(), lr=1e-4)
# exp1_optimizer_D = optim.Adam(exp1_discriminator.parameters(), lr=1e-4)

# # Experiment 2
# exp2_generator = ESRGANGenerator(num_rrdb=16, residual_scaling=0.2, init_variance=0.05).to(device)
# exp2_discriminator = ESRGANDiscriminator(num_conv_layers=7).to(device)
# exp2_optimizer_G = optim.Adam(exp2_generator.parameters(), lr=1e-4)
# exp2_optimizer_D = optim.Adam(exp2_discriminator.parameters(), lr=1e-4)

# # Experiment 3
# exp3_generator = ESRGANGenerator(num_rrdb=24, residual_scaling=0.15, init_variance=0.03).to(device)
# exp3_discriminator = ESRGANDiscriminator(num_conv_layers=12).to(device)
# exp3_optimizer_G = optim.Adam(exp3_generator.parameters(), lr=1e-4)
# exp3_optimizer_D = optim.Adam(exp3_discriminator.parameters(), lr=1e-4)

# # Experiment 4
# exp4_generator = ESRGANGenerator(num_rrdb=14, residual_scaling=0.3, init_variance=0.01).to(device)
# exp4_discriminator = ESRGANDiscriminator(num_conv_layers=8).to(device)
# exp4_optimizer_G = optim.RMSprop(exp4_generator.parameters(), lr=2e-4)
# exp4_optimizer_D = optim.RMSprop(exp4_discriminator.parameters(), lr=2e-4)

# # Experiment 5
# exp5_generator = ESRGANGenerator(num_rrdb=32, residual_scaling=0.2, init_variance=0.025).to(device)
# exp5_discriminator = ESRGANDiscriminator(num_conv_layers=9).to(device)
# exp5_optimizer_G = optim.SGD(exp5_generator.parameters(), lr=2e-4)
# exp5_optimizer_D = optim.SGD(exp5_discriminator.parameters(), lr=2e-4)

# # Experiment 6
# exp6_generator = ESRGANGenerator(num_rrdb=12, residual_scaling=0.1, init_variance=0.015).to(device)
# exp6_discriminator = ESRGANDiscriminator(num_conv_layers=28).to(device)
# exp6_optimizer_G = optim.SGD(exp6_generator.parameters(), lr=1e-4)
# exp6_optimizer_D = optim.SGD(exp6_discriminator.parameters(), lr=1e-4)

# Experiment 7
exp7_generator = ESRGANGenerator(num_rrdb=28, residual_scaling=0.15, init_variance=0.03).to(device)
exp7_discriminator = ESRGANDiscriminator(num_conv_layers=14).to(device)
exp7_optimizer_G = optim.Adam(exp7_generator.parameters(), lr=1e-4)
exp7_optimizer_D = optim.Adam(exp7_discriminator.parameters(), lr=1e-4)

# # Experiment 8
# exp8_generator = ESRGANGenerator(num_rrdb=16, residual_scaling=0.3, init_variance=0.01).to(device)
# exp8_discriminator = ESRGANDiscriminator(num_conv_layers=10).to(device)
# exp8_optimizer_G = optim.RMSprop(exp8_generator.parameters(), lr=2e-4)
# exp8_optimizer_D = optim.RMSprop(exp8_discriminator.parameters(), lr=2e-4)

# # Experiment 9
# exp9_generator = ESRGANGenerator(num_rrdb=32, residual_scaling=0.1, init_variance=0.015).to(device)
# exp9_discriminator = ESRGANDiscriminator(num_conv_layers=24).to(device)
# exp9_optimizer_G = optim.RMSprop(exp9_generator.parameters(), lr=2e-4)
# exp9_optimizer_D = optim.RMSprop(exp9_discriminator.parameters(), lr=2e-4)

# # Experiment 10
# exp10_generator = ESRGANGenerator(num_rrdb=16, residual_scaling=0.2, init_variance=0.02).to(device)
# exp10_discriminator = ESRGANDiscriminator(num_conv_layers=32).to(device)
# exp10_optimizer_G = optim.SGD(exp10_generator.parameters(), lr=1e-4)
# exp10_optimizer_D = optim.SGD(exp10_discriminator.parameters(), lr=1e-4)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim
import os

num_epochs = 100
total_adversarial_loss = 0
total_content_loss = 0

for epoch in range(num_epochs):
    real_output = discriminator(dataloader.dataset[0][0].to(device))
    fake_output = discriminator(generator(dataloader.dataset[0][0].unsqueeze(0).to(device)))
    disc_real_loss = adversarial_loss(real_output, torch.ones_like(real_output).to(device))
    disc_fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output).to(device))
    disc_loss = disc_real_loss + disc_fake_loss
    optimizer_D.zero_grad()
    disc_loss.backward()
    torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.5)
    optimizer_D.step()

    fake_output = discriminator(generator(dataloader.dataset[0][0].unsqueeze(0).to(device)))
    gen_adversarial_loss = adversarial_loss(fake_output, torch.ones_like(fake_output).to(device))
    gen_content_loss = content_loss(generator(dataloader.dataset[0][0].unsqueeze(0).to(device)), dataloader.dataset[0][0].unsqueeze(0).to(device))
    gen_loss = adversarial_loss_weight * gen_adversarial_loss + content_loss_weight * gen_content_loss
    optimizer_G.zero_grad()
    gen_loss.backward()
    torch.nn.utils.clip_grad_value_(generator.parameters(), 0.5)
    optimizer_G.step()

    total_adversarial_loss += gen_adversarial_loss.item()
    total_content_loss += gen_content_loss.item()

    if (epoch + 1) % 10 == 0:
        checkpoint_dir = "/content/models"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }, os.path.join(checkpoint_dir, f"isic_gen_e_{epoch+1}.h5"))

    # if (epoch + 1) % 10 == 0:
    #     checkpoint_dir = "/content/models"
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     torch.save({
    #         'generator': generator.state_dict(),
    #         'discriminator': discriminator.state_dict(),
    #         'optimizer_G': optimizer_G.state_dict(),
    #         'optimizer_D': optimizer_D.state_dict()
    #     }, os.path.join(checkpoint_dir, f"ham10000_gen_e_{epoch+1}.h5"))

print(f"Average Adversarial loss : {total_adversarial_loss / num_epochs:.2f}")
print(f"Average Content loss     : {total_content_loss / num_epochs:.2f}") 
