from collections import defaultdict, deque
import math
import pygame
import random
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Leak Finding Algorithm")

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
ALPHA = 0.5


class Spot:
    def __init__(self, row, col, width, total_rows) -> None:
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.buffer_neighbors = []
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

    def adj_fire(self, grid):
        adjfire = False
        # DOWN
        if self.row < self.total_rows - 1 and (grid[self.row + 1][self.col].is_fire()):
            adjfire = True
        # UP
        if self.row > 0 and (grid[self.row - 1][self.col].is_fire()):
            adjfire = True
        # RIGHT
        if self.col < self.total_rows - 1 and (grid[self.row][self.col + 1].is_fire()):
            adjfire = True
        # LEFT
        if self.col > 0 and (grid[self.row][self.col - 1].is_fire()):
            adjfire = True

        if adjfire:
            pass
        return adjfire

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


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap
    return row, col


def reconstruct_path(came_from, current, draw):
    hi = 0
    while current in came_from:
        hi += 1
        current = came_from[current]
        current.make_path()
        draw()
    return hi

# Manhattan distance from point to end


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def algorithm(draw, grid, start, end):
    count = 0

    # frontier
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    # heuristic
    g_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float('inf') for row in grid for spot in row}
    f_score[start] = g_score[start] + h(start.get_pos(), end.get_pos())

    # reached
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            length = reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            for row in grid:
                for spot in row:
                    if not (spot.is_end() or spot.is_start() or spot.is_barrier() or spot.is_path()):
                        spot.reset()
            return (True, length, came_from)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        if current != start:
            current.make_closed()

    return (False, -1, came_from)


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


def make_ship(draw, grid, rows, square):
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
        # pygame.time.delay(2000)

    # Check for deadends
    for cell in white:
        if cell.num_white_nei == 1:
            brown.add(cell)
            cell.make_color(BROWN)

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
    # pygame.time.delay(1000)

    # Left over brown made into white
    while brown:
        leftover = brown.pop()
        leftover.make_color(WHITE)

    # pygame.time.delay(1000)
    while red:
        temp = red.pop()
        temp.make_color(BLACK)

    # pygame.time.delay(1000)

    random_bot = random.choice(list(white))
    random_bot.make_start()
    x, y = random_bot.get_pos()
    detection_square_spots = set()
    for i in range((-square//2), (square//2)+1):
        for j in range((-square//2), (square//2)+1):
            if (0 <= x+i < len(grid) and 0 <= y+j < len(grid[0])):
                detection_square_spots.add(grid[x+i][y+j])

    random_leak = random.choice(list(white - detection_square_spots))
    random_leak.make_end()

    return (white, random_bot, random_leak)


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


def infinity():
    return float('inf')


def Bot3(win, width, ROWS, square):
    def check_square(spot, leak):
        x, y = spot.get_pos()
        det_square = set()
        border = set()
        ans = False
        k = square//2
        for i in range(-k-1, k+2):
            for j in range(-k-1, k+2):
                if (0 <= x+i < len(grid) and 0 <= y+j < len(grid[0])):
                    if i == -k-1 or i == k+1 or j == -k-1 or j == k+1:
                        border.add(grid[x+i][y+j])
                    else:
                        det_square.add(grid[x+i][y+j])
                        if grid[x+i][y+j] == leak:
                            ans = True
        return ans, det_square, border

    def create_dist_matrix(may_contain_leak):
        dists = defaultdict(float('inf'))

        for spot in may_contain_leak:
            for row in grid:
                for l in row:
                    if (l.is_path() or l.is_open() or l.is_closed()):
                        l.reset()
                    l.update_neighbors(grid)
                    l.update_unres_neighbors(grid)
            for row in grid:
                for l in row:
                    if l.is_white() and dists[(spot, l)] != 0:
                        _, dist, came_from = algorithm(lambda: draw(win, grid, ROWS, width),
                                                       grid, spot, l)
                        dists[(spot, l)] = dist
                        dists[(l, spot)] = dist
        return dists

    # assert square >= 3
    grid = make_grid(ROWS, width)

    may_contain_leak, random_bot, random_leak = make_ship(
        lambda: draw(win, grid, ROWS, width), grid, ROWS, square=square)

    # dists = create_dist_matrix(may_contain_leak)
    # for key, value in dists:
    #     print(type(key), type(value))

    may_contain_leak = may_contain_leak - {random_bot}

    start = random_bot
    end = random_leak

    # key is a position
    # val is a probability (float)
    probabilities = defaultdict(lambda: 1/len(may_contain_leak))
    for i in may_contain_leak:
        probabilities[i.get_pos()] = 1/len(may_contain_leak)
    # probabilities[start.get_pos()] = 0

# for each sense function, set the bot_location( current location of bot), probability of leak to 0, since we have already visited it

    def bot_enters_cell_probability_update(probability_matrix, bot_location):
        probability_matrix[bot_location] = 0
        for key in probability_matrix:
            # key is position of cell j we want to calculate updated probability for
            # key 2 is position of every other cell j', used for summation stored in denom
            denom = sum(
                probability_matrix[key2] for key2 in probability_matrix if key2 != bot_location)
            # if denom != 0 and not math.isinf(denom):
            probability_matrix[key] = probability_matrix[key] / denom
        return probability_matrix

    def beep_probability_update(probability_matrix, bot_location):
        probability_matrix[bot_location] = 0
        for key in probability_matrix:
            denom = sum(
                probability_matrix[key2] *
                math.exp((-1 * ALPHA) * (dists[(bot_location, key2)] - 1))
                for key2 in probability_matrix
                if key2 != bot_location
            )
            if denom != 0 and not math.isinf(denom):
                probability_matrix[key] = (
                    probability_matrix[key] *
                    math.exp((-1 * ALPHA) * (dists[(bot_location, key)] - 1))
                ) / denom

        return probability_matrix

    def no_beep_probability_update(probability_matrix, bot_location):
        probability_matrix[bot_location] = 0
        for key in probability_matrix:
            denom = sum(
                probability_matrix[key2] *
                (1 - math.exp((-1 * ALPHA) *
                 (dists[(bot_location, key2)] - 1)))
                for key2 in probability_matrix
                if key2 != bot_location
            )
            if denom != 0 and not math.isinf(denom):

                probability_matrix[key] = (
                    probability_matrix[key] * (1 - math.exp((-1 * ALPHA)
                                               * (dists[(bot_location, key)] - 1)))
                ) / denom
        return probability_matrix

    def get_location_of_max_probability(probability_matrix):
        # returns key of max probability
        (row, col) = max(probability_matrix, key=probability_matrix.get)
        return grid[row][col]

    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)

    run = True
    time = True
    total_actions = 0

    while run:
        for row in grid:
            for spot in row:
                spot.update_neighbors(grid)
                spot.update_unres_neighbors(grid)
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if time:

            # pseudocode: while bot_location != leak_location:
            while (start.get_pos() != random_leak.get_pos()):
                # for _ in range(100):
                # print(sum(probabilities.values()))
                queue = deque()
                dists = defaultdict(infinity)

                for og_nei in may_contain_leak:
                    queue.append(og_nei)
                    dists[(og_nei.get_pos(), og_nei.get_pos())] = 0
                    while queue:

                        curr = queue.popleft()

                        for nei in curr.neighbors:
                            if dists[(og_nei.get_pos(), nei.get_pos())] != float('inf'):
                                continue
                            else:
                                dists[(og_nei.get_pos(), nei.get_pos())] = dists[(
                                    og_nei.get_pos(), curr.get_pos())]+1
                                dists[(nei.get_pos(), og_nei.get_pos())] = dists[(
                                    og_nei.get_pos(), curr.get_pos())]

                                queue.append(nei)
                # probability of hearing beep in cell bot_location due to leak in leak_location
                # print(dists)
                beep = random.random() <= (
                    math.exp((-1*ALPHA)*(dists[start.get_pos(), random_leak.get_pos()] - 1)))
                # print("beep: ", beep, math.exp((-1*ALPHA) *(dists[start.get_pos(), random_leak.get_pos()] - 1))),
                # Run Sense
                # removed all leak_present code here as no detection square for bot 3
                total_actions += 1
                if beep:
                    probabilities = beep_probability_update(
                        probabilities, start.get_pos())
                else:
                    probabilities = no_beep_probability_update(
                        probabilities, start.get_pos())

                # Find next spot to explore
                sense_again = all(not i.is_path() for i in start.neighbors)
                if sense_again or beep:

                    next_location = get_location_of_max_probability(
                        probabilities)
                    # print(probabilities)

                    a, temp, came_from = algorithm(lambda: draw(win, grid, ROWS, width),
                                                   grid, start, next_location)
                    total_actions += temp
                    # print(len(came_from))

                # get path from bot location to the next location found

                random_leak.make_color(BROWN)

                # while i!= next_location:
                for i in start.neighbors:
                    if i.get_pos() == random_leak.get_pos():
                        return total_actions
                    else:
                        if i.is_path() or i.get_pos() == next_location.get_pos():
                            # print(i.get_pos())
                            # print(i.get_pos(),probabilities)
                            # pygame.time.delay(100)

                            i.make_start()
                            # may_contain_leak = may_contain_leak - {i}
                            start.reset()
                            start = i
                            # print(start.get_pos())
                            if start.get_pos() == random_leak.get_pos():
                                return total_actions
                            else:
                                probabilities = bot_enters_cell_probability_update(
                                    probabilities, start.get_pos())
                                # print("reached")
                                probabilities[start.get_pos()] = 0
                # pygame.time.delay(1000)

                draw(win, grid, ROWS, width)
            time = False

    pygame.quit()
    return total_actions


def main(win, width):
    ROWS = 30
    # make them return FAILED OR SUCCEEDED, ALSO PASS IN Q
    actions = Bot3(win, width,  ROWS, 3)
    print(actions)


main(WIN, WIDTH)
