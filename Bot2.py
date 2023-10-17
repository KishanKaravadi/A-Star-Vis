import pygame
import math
import random
from queue import PriorityQueue

WIDTH = 800
Q = 0.3
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREEN_YELLOW = (60, 214, 111)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
BROWN = (79, 46, 13)


# D = 3

class Spot:
    def __init__(self, row, col, width, total_rows) -> None:
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.unres_neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.num_white_nei = 0
        # self.path = False
        self.fire = False

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_white(self):
        return self.color == WHITE

    def is_start(self):
        return self.color == TURQUOISE

    def is_end(self):
        return self.color == GREEN_YELLOW

    def is_fire(self):
        return self.fire

    def is_path(self):
        return self.color == PURPLE

    def reset(self):
        self.color = WHITE
        # self.path = False

    def make_color(self, color) -> None:
        self.color = color

    def get_color(self):
        return self.color

    def make_start(self):
        self.color = TURQUOISE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_fire(self):
        self.fire = True

    def make_end(self):
        self.color = GREEN_YELLOW

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))
        if (self.fire):
            pygame.draw.rect(
                win, ORANGE, (self.x, self.y, self.width/2, self.width/2))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not (grid[self.row + 1][self.col].is_barrier() or grid[self.row + 1][self.col].is_fire()):
            self.neighbors.append(grid[self.row + 1][self.col])
        # UP
        if self.row > 0 and not (grid[self.row - 1][self.col].is_barrier() or grid[self.row - 1][self.col].is_fire()):
            self.neighbors.append(grid[self.row - 1][self.col])
        # RIGHT
        if self.col < self.total_rows - 1 and not (grid[self.row][self.col + 1].is_barrier() or grid[self.row][self.col + 1].is_fire()):
            self.neighbors.append(grid[self.row][self.col + 1])
        # LEFT
        if self.col > 0 and not (grid[self.row][self.col - 1].is_barrier() or grid[self.row][self.col - 1].is_fire()):
            self.neighbors.append(grid[self.row][self.col - 1])

    def update_unres_neighbors(self, grid):
        self.unres_neighbors = []
        # DOWN
        if self.row < self.total_rows - 1:
            self.unres_neighbors.append(grid[self.row + 1][self.col])
        # UP
        if self.row > 0:
            self.unres_neighbors.append(grid[self.row - 1][self.col])
        # RIGHT
        if self.col < self.total_rows - 1:
            self.unres_neighbors.append(grid[self.row][self.col + 1])
        # LEFT
        if self.col > 0:
            self.unres_neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def h_time(p1, p2, time, rows):
    x1, y1 = p1
    x2, y2 = p2
    # return 1/((abs(x1 - x2) + abs(y1 - y2)) / (1+time))*50000
    return 1/((abs(x1 - x2) + abs(y1 - y2) + 1))*rows*100


def algorithm(draw, grid, start, end, fire, time, rows):
    count = 0

    # frontier
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    # heuristic
    g_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float('inf') for row in grid for spot in row}
    f_score[start] = g_score[start] + \
        h_time(start.get_pos(), fire.get_pos(), time, rows=rows)

    # reached
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            for row in grid:
                for spot in row:
                    if not (spot.is_end() or spot.is_start() or spot.is_barrier() or spot.is_path()):
                        spot.reset()
                        draw()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h_time(neighbor.get_pos(), fire.get_pos(), time, rows)
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()

    start.make_start()
    end.make_end()
    for row in grid:
        for spot in row:
            if not (spot.is_end() or spot.is_start() or spot.is_barrier() or spot.is_path()):
                spot.reset()
                draw()
    return False


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap
    return row, col


def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            spot.make_barrier()
            grid[i].append(spot)
    return grid


def make_ship(draw, grid, rows):
    # Find a random spot on the grid
    random_row = random.randint(0, rows-1)
    random_col = random.randint(0, rows-1)

    start = grid[random_row][random_col]

    start.make_open()

    # Set of neighbors
    white = set()
    green = {start}
    red = set()
    brown = set()

    while green:
        # Pop the current one
        curr_cell = green.pop()

        # Turn it white
        curr_cell.make_color(WHITE)
        white.add(curr_cell)

        # Look at neighbors
        curr_cell.update_unres_neighbors(grid)

        for cell in curr_cell.unres_neighbors:
            cell.num_white_nei += 1

        # Make ship without deadends
        for cell in curr_cell.unres_neighbors:
            if cell.color == BLACK:
                cell.make_color(GREEN)
                green.add(cell)
            elif cell.color == GREEN:
                cell.make_color(RED)
                green.remove(cell)
                red.add(cell)
        draw()
        # pygame.time.delay(2000)

    # Check for deadends
    for cell in white:
        # print((cell.row, cell.col, cell.num_white_nei))
        if cell.num_white_nei == 1:
            brown.add(cell)
            cell.make_color(BROWN)

        draw()
        # pygame.time.delay(2000)

    # Half of the deadends get checked and made into cycles
    for _ in range(len(brown)//2):
        deadend = brown.pop()
        for nei in deadend.unres_neighbors:
            if nei.color == RED:
                red.remove(nei)
                nei.make_color(WHITE)
                white.add(nei)
                for neinei in nei.unres_neighbors:
                    neinei.num_white_nei += 1
                break
        deadend.make_color(WHITE)
        # pygame.time.delay(2000)
    draw()
    # pygame.time.delay(1000)

    # Left over brown made into white
    while brown:
        leftover = brown.pop()
        leftover.make_color(WHITE)

    draw()
    # pygame.time.delay(1000)
    while red:
        temp = red.pop()
        temp.make_color(BLACK)

    # pygame.time.delay(1000)
    draw()

    random_bot = random.choice(list(white))
    random_bot.make_start()

    random_button = random.choice(list(white - {random_bot}))
    random_button.make_end()

    random_fire = random.choice(list(white - {random_bot, random_button}))
    random_fire.make_fire()

    draw()

    return (random_bot, random_button, random_fire)


def draw_grid_lines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid_lines(win, rows, width)
    pygame.display.update()


def main(win, width):
    ROWS = 100
    grid = make_grid(ROWS, width)
    random_bot, random_button, random_fire = make_ship(
        lambda: draw(win, grid, ROWS, width), grid, ROWS)

    start = random_bot
    end = random_button
    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)
    print(random_bot.get_pos())
    timeStep = 0
    time = False
    run = True

    # spot.update_unres_neighbors(grid)
    a = algorithm(lambda: draw(win, grid, ROWS, width),
                  grid, start, end, fire=random_fire, time=timeStep, rows=ROWS)

    while run:

        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # Left Mouse Button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_fire()

            elif pygame.mouse.get_pressed()[2]:  # Right Mouse Button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.fire = False
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
            elif pygame.mouse.get_pressed()[1]:  # Middle Mouse Button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.make_color(BLACK)
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
            elif event.type == pygame.KEYDOWN:  # Middle Mouse Button
                if event.key == pygame.K_d:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    spot.make_color(RED)
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            if event.type == pygame.KEYDOWN:  # Space
                if event.key == pygame.K_SPACE and start and end:
                    time = True

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
        if (time):

            timeStep += 100000

            for row in grid:
                for spot in row:
                    if (spot.is_path() or spot.is_open() or spot.is_closed()):
                        spot.reset()
                    spot.update_neighbors(grid)
                    spot.update_unres_neighbors(grid)

            a = algorithm(lambda: draw(win, grid, ROWS, width),
                          grid, start, end, fire=random_fire, time=timeStep, rows=ROWS)

            if not (a):
                time = False
                print("NO PATHS AVAILABLE")
                continue

            # pygame.time.delay(2000)
            print("ONE TIME STEP PASSED")
            for nei in start.neighbors:
                print(nei.get_pos(), nei.get_color(), nei.is_fire())
                if nei.is_path() or nei.is_end():
                    print("YES")
                    nei.make_start()
                    start.reset()
                    start = nei
            if (start == end):
                print("HOORAYYYY")
                # pygame.time.delay(300)
                time = False
                continue
            firespots = set()
            for row in grid:
                for spot in row:
                    if (spot.is_white() or spot.is_start() or spot.is_end() or spot.is_path()):
                        K = 0
                        for nei in spot.unres_neighbors:
                            if nei.is_fire():
                                K += 1
                        # if(K > 0):
                        #     print(K, "SKDUJFGHSD")
                        probability = 1 - ((1 - Q)**K)

                        # print(probability)
                        if random.random() < probability:
                            firespots.add(spot)
                            # print("added")
                        else:
                            pass
            for spot in firespots:
                spot.make_fire()
            if (start.is_fire() or end.is_fire()):
                print("RIP BOZO")
                # pygame.time.delay(300)
                time = False
                continue

    pygame.quit()


main(WIN, WIDTH)
