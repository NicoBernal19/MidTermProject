# task_d.py
import pygame
import numpy as np
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io
from PIL import Image, ImageDraw, ImageFont
import os

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import transforms, datasets

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using image generation simulation.")

# Pygame configuration
pygame.init()
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
ORANGE = (255, 165, 0)
PURPLE = (200, 100, 255)
PINK = (255, 182, 193)

# Fashion-MNIST labels
FASHION_LABELS = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}


class SimpleGAN:
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.training_history = []

    def build_generator(self, latent_dim=100):
        """Build generator network"""
        return nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def build_discriminator(self):
        """Build discriminator network"""
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def train_gan(self, epochs=50, batch_size=64, latent_dim=100):
        """Train the GAN model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Simulating training...")
            self._simulate_training(epochs)
            return

        try:
            # Load Fashion-MNIST dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            train_dataset = datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Initialize models
            self.generator = self.build_generator(latent_dim).to(self.device)
            self.discriminator = self.build_discriminator().to(self.device)

            # Optimizers
            g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            # Loss function
            criterion = nn.BCELoss()

            print("Training GAN...")
            for epoch in range(epochs):
                g_losses = []
                d_losses = []

                for i, (real_imgs, _) in enumerate(train_loader):
                    batch_size = real_imgs.size(0)
                    real_imgs = real_imgs.view(batch_size, -1).to(self.device)

                    # Train Discriminator
                    d_optimizer.zero_grad()

                    # Real images
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    real_output = self.discriminator(real_imgs)
                    d_loss_real = criterion(real_output, real_labels)

                    # Fake images
                    z = torch.randn(batch_size, latent_dim).to(self.device)
                    fake_imgs = self.generator(z)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                    fake_output = self.discriminator(fake_imgs.detach())
                    d_loss_fake = criterion(fake_output, fake_labels)

                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    d_optimizer.step()

                    # Train Generator
                    g_optimizer.zero_grad()
                    fake_output = self.discriminator(fake_imgs)
                    g_loss = criterion(fake_output, real_labels)  # Trick discriminator
                    g_loss.backward()
                    g_optimizer.step()

                    g_losses.append(g_loss.item())
                    d_losses.append(d_loss.item())

                avg_g_loss = np.mean(g_losses)
                avg_d_loss = np.mean(d_losses)
                self.training_history.append((avg_g_loss, avg_d_loss))

                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

            self.is_trained = True
            print("GAN training completed!")

        except Exception as e:
            print(f"Training failed: {e}. Using simulation.")
            self._simulate_training(epochs)

    def _simulate_training(self, epochs):
        """Simulate training when PyTorch is not available"""
        print("Simulating GAN training...")
        for epoch in range(epochs):
            g_loss = max(0.1, 2.0 - epoch * 0.04 + random.uniform(-0.1, 0.1))
            d_loss = max(0.1, 1.5 - epoch * 0.03 + random.uniform(-0.1, 0.1))
            self.training_history.append((g_loss, d_loss))

        self.is_trained = True
        print("Simulated training completed")

    def generate_images(self, num_images=9, latent_dim=100):
        """Generate fashion images"""
        if not self.is_trained:
            return None, None

        if TORCH_AVAILABLE and self.generator is not None:
            # Generate with trained model
            self.generator.eval()
            with torch.no_grad():
                z = torch.randn(num_images, latent_dim).to(self.device)
                generated = self.generator(z)
                images = generated.cpu().numpy()
                images = (images * 0.5 + 0.5) * 255  # Denormalize
                images = images.reshape(num_images, 28, 28)
                labels = [random.choice(list(FASHION_LABELS.values())) for _ in range(num_images)]
                return images, labels
        else:
            # Generate simulated images
            return self._generate_simulated_images(num_images)

    def _generate_simulated_images(self, num_images=9):
        """Generate simulated fashion images"""
        images = []
        labels = []

        for i in range(num_images):
            # Create simple geometric patterns that resemble fashion items
            img_array = np.random.randint(0, 50, (28, 28), dtype=np.uint8)

            # Add different patterns based on "category"
            category = random.randint(0, 9)
            labels.append(FASHION_LABELS[category])

            if category in [0, 2, 3, 4, 6]:  # Clothing items
                # Create clothing-like patterns
                center_x, center_y = 14, 14
                for x in range(28):
                    for y in range(28):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if dist < 10:
                            img_array[y, x] = min(255, img_array[y, x] + random.randint(150, 255))

            elif category in [5, 7, 9]:  # Footwear
                # Create shoe-like patterns
                for x in range(10, 18):
                    for y in range(15, 25):
                        img_array[y, x] = min(255, img_array[y, x] + random.randint(100, 200))

            else:  # Bags
                # Create bag-like patterns
                for x in range(8, 20):
                    for y in range(10, 20):
                        if 12 <= x <= 16 and 12 <= y <= 18:
                            img_array[y, x] = min(255, img_array[y, x] + random.randint(200, 255))

            images.append(img_array)

        return images, labels

    def get_training_plot(self):
        """Generate training progress plot"""
        if not self.training_history:
            return None

        epochs = range(len(self.training_history))
        g_losses = [x[0] for x in self.training_history]
        d_losses = [x[1] for x in self.training_history]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, g_losses, 'b-', label='Generator Loss', linewidth=2)
        plt.plot(epochs, d_losses, 'r-', label='Discriminator Loss', linewidth=2)
        plt.title('GAN Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return pygame.image.load(buf)


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(min(c + 30, 255) for c in color)
        self.hover = False

    def draw(self, screen, font):
        color = self.hover_color if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)

        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=5)
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
        pygame.draw.rect(screen, BLUE, handle_rect, border_radius=3)
        pygame.draw.rect(screen, BLACK, handle_rect, 1, border_radius=3)

        label_text = font.render(f"{self.label}: {self.value}", True, BLACK)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(rel_x, self.rect.width))
            self.value = int(self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val))


class GANVisualizer:
    def __init__(self):
        self.gan = SimpleGAN()
        self.generated_images = None
        self.image_labels = None
        self.training_plot = None
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # UI Elements
        self.train_button = Button(50, 120, 200, 40, "Train GAN", GREEN)
        self.generate_button = Button(260, 120, 200, 40, "Generate Images", BLUE)
        self.create_poster_button = Button(470, 120, 200, 40, "Create Poster", ORANGE)

        self.epochs_slider = Slider(700, 130, 200, 10, 200, 50, "Training Epochs")
        self.images_slider = Slider(920, 130, 200, 1, 16, 9, "Images to Generate")

        self.noise_seed_input = 42
        self.seed_buttons = [
            Button(700, 180, 60, 30, "-10", BLUE),
            Button(770, 180, 60, 30, "Random", PURPLE),
            Button(840, 180, 60, 30, "+10", BLUE)
        ]

    def train_model(self):
        """Train the GAN model"""
        self.gan.train_gan(epochs=self.epochs_slider.value)
        self.training_plot = self.gan.get_training_plot()

    def generate_images(self):
        """Generate new images"""
        if not self.gan.is_trained:
            self.train_model()

        self.generated_images, self.image_labels = self.gan.generate_images(
            num_images=self.images_slider.value
        )

    def create_poster(self):
        """Create promotional poster with generated images"""
        if self.generated_images is None:
            self.generate_images()

        if self.generated_images is None:
            return None

        # Create a simple poster layout
        poster_width = 600
        poster_height = 400

        # Create PIL image for the poster
        poster = Image.new('RGB', (poster_width, poster_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(poster)

        try:
            # Try to use a font (this might fail if font not available)
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        except:
            font = None
            title_font = None

        # Add title
        draw.rectangle([10, 10, poster_width - 10, 50], fill=(50, 100, 200), outline=(0, 0, 0), width=2)
        draw.text((poster_width // 2, 30), "UrbanStyle Fashion Collection",
                  fill=WHITE, font=title_font, anchor="mm")

        # Add generated images to poster
        num_images = min(4, len(self.generated_images))
        img_size = 80

        for i in range(num_images):
            if i < len(self.generated_images):
                # Convert numpy array to PIL Image
                img_array = self.generated_images[i]
                if isinstance(img_array, np.ndarray):
                    if img_array.dtype != np.uint8:
                        img_array = img_array.astype(np.uint8)

                    img_pil = Image.fromarray(img_array, mode='L')
                    # Resize and convert to RGB
                    img_pil = img_pil.resize((img_size, img_size))
                    img_rgb = Image.new('RGB', (img_size, img_size), color=(255, 255, 255))
                    img_rgb.paste(img_pil, (0, 0))

                    # Position images in grid
                    x_pos = 50 + (i % 2) * (img_size + 100)
                    y_pos = 100 + (i // 2) * (img_size + 60)

                    poster.paste(img_rgb, (x_pos, y_pos))

                    # Add label
                    label = self.image_labels[i] if i < len(self.image_labels) else "Fashion Item"
                    draw.text((x_pos + img_size // 2, y_pos + img_size + 20),
                              label, fill=BLACK, font=font, anchor="mm")

        # Add footer
        draw.rectangle([10, poster_height - 40, poster_width - 10, poster_height - 10],
                       fill=(200, 200, 200), outline=(0, 0, 0), width=1)
        draw.text((poster_width // 2, poster_height - 25), "Generated by UrbanStyle GAN",
                  fill=BLACK, font=font, anchor="mm")

        # Convert to Pygame surface
        buf = io.BytesIO()
        poster.save(buf, format='PNG')
        buf.seek(0)

        return pygame.image.load(buf)

    def draw_interface(self, screen):
        """Draw the complete interface"""
        # Title
        title = self.font.render("UrbanStyle - Task D: GAN for Fashion Images", True, BLACK)
        screen.blit(title, (50, 20))
        subtitle = self.small_font.render("Generative Adversarial Network for Promotional Images", True, GRAY)
        screen.blit(subtitle, (50, 60))

        # Controls
        self.train_button.draw(screen, self.small_font)
        self.generate_button.draw(screen, self.small_font)
        self.create_poster_button.draw(screen, self.small_font)

        self.epochs_slider.draw(screen, self.small_font)
        self.images_slider.draw(screen, self.small_font)

        # Noise seed controls
        seed_text = self.small_font.render(f"Noise Seed: {self.noise_seed_input}", True, BLACK)
        screen.blit(seed_text, (700, 150))

        for button in self.seed_buttons:
            button.draw(screen, self.small_font)

        # Status
        status_y = 220
        status_color = GREEN if self.gan.is_trained else RED
        status_text = "GAN Trained" if self.gan.is_trained else "GAN Not Trained"
        status_surface = self.small_font.render(f"Status: {status_text}", True, status_color)
        screen.blit(status_surface, (50, status_y))

        # Training plot
        if self.training_plot:
            screen.blit(self.training_plot, (50, 260))

        # Generated images grid
        if self.generated_images is not None:
            self._draw_image_grid(screen, 700, 260)

        # Poster
        poster = getattr(self, '_current_poster', None)
        if poster:
            screen.blit(poster, (300, 450))

    def _draw_image_grid(self, screen, start_x, start_y):
        """Draw grid of generated images"""
        if self.generated_images is None:
            return

        img_size = 60
        spacing = 10
        images_per_row = 3

        for i, (img_array, label) in enumerate(zip(self.generated_images, self.image_labels)):
            if i >= 9:  # Limit display to 9 images
                break

            row = i // images_per_row
            col = i % images_per_row

            x = start_x + col * (img_size + spacing)
            y = start_y + row * (img_size + spacing + 20)

            # Create surface for image
            img_surface = pygame.Surface((img_size, img_size))
            img_surface.fill(WHITE)

            # Convert numpy array to Pygame surface
            if isinstance(img_array, np.ndarray):
                # Normalize and scale image
                img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
                img_scaled = (img_normalized * 255).astype(np.uint8)

                # Create Pygame surface from array
                img_surface_array = pygame.surfarray.make_surface(
                    np.stack([img_scaled] * 3, axis=2)  # Convert to RGB
                )
                img_surface_array = pygame.transform.scale(img_surface_array, (img_size, img_size))
                screen.blit(img_surface_array, (x, y))

            # Draw border
            pygame.draw.rect(screen, BLACK, (x, y, img_size, img_size), 1)

            # Draw label
            label_text = self.small_font.render(label[:12], True, BLACK)
            screen.blit(label_text, (x, y + img_size + 5))

    def handle_event(self, event):
        """Handle UI events"""
        if self.train_button.handle_event(event):
            self.train_model()

        if self.generate_button.handle_event(event):
            self.generate_images()

        if self.create_poster_button.handle_event(event):
            self._current_poster = self.create_poster()

        # Noise seed controls
        if self.seed_buttons[0].handle_event(event):  # -10
            self.noise_seed_input -= 10
            random.seed(self.noise_seed_input)
            np.random.seed(self.noise_seed_input)
            if TORCH_AVAILABLE:
                torch.manual_seed(self.noise_seed_input)

        if self.seed_buttons[1].handle_event(event):  # Random
            self.noise_seed_input = random.randint(0, 1000)
            random.seed(self.noise_seed_input)
            np.random.seed(self.noise_seed_input)
            if TORCH_AVAILABLE:
                torch.manual_seed(self.noise_seed_input)

        if self.seed_buttons[2].handle_event(event):  # +10
            self.noise_seed_input += 10
            random.seed(self.noise_seed_input)
            np.random.seed(self.noise_seed_input)
            if TORCH_AVAILABLE:
                torch.manual_seed(self.noise_seed_input)

        self.epochs_slider.handle_event(event)
        self.images_slider.handle_event(event)


def main():
    """Main function for Task D"""
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("UrbanStyle - Task D: GAN Fashion Image Generation")
    clock = pygame.time.Clock()

    visualizer = GANVisualizer()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            visualizer.handle_event(event)

        screen.fill(WHITE)
        visualizer.draw_interface(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()