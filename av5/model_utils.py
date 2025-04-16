import torch
import matplotlib.pyplot as plt


def sample_images(generator):
    generator.eval()
    noise = torch.randn(1, 100).to('cuda')
    gen_img = generator(noise)
    gen_img = gen_img.view(1, 3, 32, 32)
    gen_img = gen_img * 0.5 + 0.5
    plt.imshow(gen_img[0].detach().cpu().permute(1, 2, 0).numpy())
    plt.show()


def train(generator, discriminator, data_loader,
          optimizer_generator, optimizer_discriminator, criterion,
          latent_dim=100, epochs=5):
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size_curr = imgs.size(0)

            valid = torch.ones(batch_size_curr, 1).to('cuda')
            fake = torch.zeros(batch_size_curr, 1).to('cuda')

            real_imgs = imgs.to('cuda')

            optimizer_generator.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim).to('cuda')
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_generator.step()

            optimizer_discriminator.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_discriminator.step()

            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % 5 == 0:
            sample_images(generator)
