import pygame
import random
from collections import deque
import heapq
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
GRID_ROWS = 6
GRID_COLS = 8
CELL_SIZE = 80
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
BLUE = (100, 150, 255)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
PURPLE = (200, 100, 255)
PINK = (255, 182, 193)

# Product categories and their sections
PRODUCT_SECTIONS = {
    "Entrance": [(0, 3), (0, 4)],
    "Men's Clothing": [(1, 1), (1, 2), (2, 1), (2, 2)],
    "Women's Clothing": [(1, 5), (1, 6), (2, 5), (2, 6)],
    "Accessories": [(3, 2), (3, 3), (3, 4), (3, 5)],
    "Shoes": [(4, 1), (4, 2), (5, 1), (5, 2)],
    "Beauty": [(4, 5), (4, 6), (5, 5), (5, 6)],
    "Checkout": [(5, 3), (5, 4)]
}

SECTION_COLORS = {
    "Entrance": GREEN,
    "Men's Clothing": BLUE,
    "Women's Clothing": PINK,
    "Accessories": ORANGE,
    "Shoes": PURPLE,
    "Beauty": YELLOW,
    "Checkout": RED
}


class Customer:
    def __init__(self, start_pos, target_sections, customer_id):
        self.pos = list(start_pos)
        self.target_sections = target_sections
        self.current_target_idx = 0
        self.path = []
        self.customer_id = customer_id
        self.speed = 3  # frames per move
        self.frame_count = 0
        self.finished = False

    def update(self):
        if self.finished:
            return

        self.frame_count += 1
        if self.frame_count < self.speed:
            return

        self.frame_count = 0

        if self.path:
            next_pos = self.path.pop(0)
            self.pos = list(next_pos)

        if not self.path and self.current_target_idx < len(self.target_sections):
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.target_sections):
                self.finished = True


class Store:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.traffic_map = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.customers = []
        self.customer_counter = 0
        self.algorithm = "BFS"  # BFS or A*

    def get_section_cells(self, section_name):
        return PRODUCT_SECTIONS.get(section_name, [])

    def get_cell_section(self, row, col):
        for section, cells in PRODUCT_SECTIONS.items():
            if (row, col) in cells:
                return section
        return None

    def is_valid_cell(self, row, col):
        return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS

    def get_neighbors(self, row, col):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_cell(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

    def bfs(self, start, goal):
        """Breadth-First Search"""
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (row, col), path = queue.popleft()

            if (row, col) == goal:
                return path

            for neighbor in self.get_neighbors(row, col):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return [start]

    def heuristic(self, pos1, pos2):
        """Manhattan distance heuristic for A*"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def astar(self, start, goal):
        """A* Search Algorithm"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current[0], current[1]):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [start]

    def find_path(self, start, goal):
        """Find path using selected algorithm"""
        if self.algorithm == "BFS":
            return self.bfs(start, goal)
        else:
            return self.astar(start, goal)

    def add_customer(self):
        """Add a new customer with random shopping targets"""
        entrance_cells = PRODUCT_SECTIONS["Entrance"]
        start_pos = random.choice(entrance_cells)

        # Random shopping journey: 2-4 sections + checkout
        all_sections = ["Men's Clothing", "Women's Clothing", "Accessories", "Shoes", "Beauty"]
        num_sections = random.randint(2, 4)
        target_section_names = random.sample(all_sections, num_sections) + ["Checkout"]

        target_sections = []
        for section_name in target_section_names:
            cells = PRODUCT_SECTIONS[section_name]
            target_sections.append(random.choice(cells))

        customer = Customer(start_pos, target_sections, self.customer_counter)
        self.customer_counter += 1

        # Calculate initial path
        if target_sections:
            path = self.find_path(tuple(customer.pos), target_sections[0])
            customer.path = path[1:]  # exclude starting position

        self.customers.append(customer)

    def update_customers(self):
        """Update all customers and recalculate paths if needed"""
        for customer in self.customers[:]:
            if customer.finished:
                self.customers.remove(customer)
                continue

            # Update traffic map
            self.traffic_map[customer.pos[0]][customer.pos[1]] += 1

            # If customer reached current target, calculate path to next
            if not customer.path and customer.current_target_idx < len(customer.target_sections):
                next_target = customer.target_sections[customer.current_target_idx]
                path = self.find_path(tuple(customer.pos), next_target)
                customer.path = path[1:]

            customer.update()

    def reset_traffic(self):
        """Reset traffic map"""
        self.traffic_map = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]


class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False

    def draw(self, screen, font):
        color = tuple(min(c + 30, 255) for c in self.color) if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)

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
        # Draw track
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=5)

        # Draw handle
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
        pygame.draw.rect(screen, BLUE, handle_rect, border_radius=3)

        # Draw label and value
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


def draw_grid(screen, store, show_heatmap):
    """Draw the store grid with sections and heatmap"""
    font = pygame.font.Font(None, 20)

    # Calculate heatmap max for normalization
    max_traffic = max(max(row) for row in store.traffic_map) if show_heatmap else 1
    max_traffic = max(max_traffic, 1)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = GRID_OFFSET_X + col * CELL_SIZE
            y = GRID_OFFSET_Y + row * CELL_SIZE

            # Get section color
            section = store.get_cell_section(row, col)
            base_color = SECTION_COLORS.get(section, WHITE) if section else WHITE

            # Apply heatmap overlay
            if show_heatmap and store.traffic_map[row][col] > 0:
                intensity = min(store.traffic_map[row][col] / max_traffic, 1.0)
                overlay_color = (255, int(255 * (1 - intensity)), int(255 * (1 - intensity)))
                # Blend colors
                base_color = tuple(int(b * 0.5 + o * 0.5) for b, o in zip(base_color, overlay_color))

            pygame.draw.rect(screen, base_color, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, DARK_GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

            # Draw section label
            if section and col == PRODUCT_SECTIONS[section][0][1] and row == PRODUCT_SECTIONS[section][0][0]:
                text = font.render(section.split()[0], True, BLACK)
                screen.blit(text, (x + 5, y + 5))


def draw_customers(screen, store):
    """Draw all customers on the grid"""
    for customer in store.customers:
        x = GRID_OFFSET_X + customer.pos[1] * CELL_SIZE + CELL_SIZE // 2
        y = GRID_OFFSET_Y + customer.pos[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 8)
        pygame.draw.circle(screen, BLACK, (x, y), 8, 2)


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("UrbanStyle - Task A: Customer Pathfinding")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)

    store = Store()
    show_heatmap = False

    # UI Elements
    add_customer_btn = Button(700, 120, 150, 40, "Add Customer", GREEN)
    clear_btn = Button(870, 120, 150, 40, "Clear All", RED)
    toggle_algorithm_btn = Button(700, 180, 150, 40, "Algorithm: BFS", BLUE)
    toggle_heatmap_btn = Button(870, 180, 150, 40, "Heatmap: OFF", ORANGE)
    reset_traffic_btn = Button(700, 240, 320, 40, "Reset Traffic Data", PURPLE)

    customer_slider = Slider(700, 320, 320, 1, 20, 5, "Customers to Add")
    speed_slider = Slider(700, 380, 320, 1, 10, 3, "Customer Speed")

    # Stats
    frame_count = 0
    auto_add_interval = 60

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle buttons
            if add_customer_btn.handle_event(event):
                for _ in range(customer_slider.value):
                    store.add_customer()

            if clear_btn.handle_event(event):
                store.customers.clear()

            if toggle_algorithm_btn.handle_event(event):
                store.algorithm = "A*" if store.algorithm == "BFS" else "BFS"
                toggle_algorithm_btn.text = f"Algorithm: {store.algorithm}"

            if toggle_heatmap_btn.handle_event(event):
                show_heatmap = not show_heatmap
                toggle_heatmap_btn.text = f"Heatmap: {'ON' if show_heatmap else 'OFF'}"

            if reset_traffic_btn.handle_event(event):
                store.reset_traffic()

            # Handle sliders
            customer_slider.handle_event(event)
            speed_slider.handle_event(event)

        # Update customer speed
        for customer in store.customers:
            customer.speed = speed_slider.value

        # Update simulation
        store.update_customers()

        # Draw everything
        screen.fill(WHITE)

        # Title
        title = font.render("UrbanStyle Store - Customer Pathfinding Simulation", True, BLACK)
        screen.blit(title, (50, 20))

        # Draw grid and customers
        draw_grid(screen, store, show_heatmap)
        draw_customers(screen, store)

        # Draw UI elements
        add_customer_btn.draw(screen, small_font)
        clear_btn.draw(screen, small_font)
        toggle_algorithm_btn.draw(screen, small_font)
        toggle_heatmap_btn.draw(screen, small_font)
        reset_traffic_btn.draw(screen, small_font)
        customer_slider.draw(screen, small_font)
        speed_slider.draw(screen, small_font)

        # Draw stats
        stats_y = 450
        stats = [
            f"Active Customers: {len(store.customers)}",
            f"Total Customers Served: {store.customer_counter}",
            f"Algorithm: {store.algorithm}",
            f"Heatmap: {'ON' if show_heatmap else 'OFF'}"
        ]

        for i, stat in enumerate(stats):
            stat_text = small_font.render(stat, True, BLACK)
            screen.blit(stat_text, (700, stats_y + i * 30))

        # Draw legend
        legend_y = 580
        legend_title = small_font.render("Store Sections:", True, BLACK)
        screen.blit(legend_title, (700, legend_y))

        legend_items = list(SECTION_COLORS.items())[:7]
        for i, (section, color) in enumerate(legend_items):
            y_pos = legend_y + 25 + (i % 4) * 25
            x_pos = 700 if i < 4 else 900
            pygame.draw.rect(screen, color, (x_pos, y_pos, 20, 20))
            pygame.draw.rect(screen, BLACK, (x_pos, y_pos, 20, 20), 1)
            text = small_font.render(section, True, BLACK)
            screen.blit(text, (x_pos + 25, y_pos))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()