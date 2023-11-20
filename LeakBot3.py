from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from collections import defaultdict, deque
import math
# import pygame
import random
from queue import PriorityQueue
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import traceback
from multiprocessing import Pool
import gc


E = math.e
FPS = 60
# clock = pygame.time.Clock()

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


class Spot:
    def __init__(self, row, col, width, total_rows) -> None:
        """
        Initialize a Spot object representing a cell in the grid.

        Args:
            row: Row index of the cell.
            col: Column index of the cell.
            width: Width of the cell in the grid.
        """
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
        # draw()# Here
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
    came_to = {}

    # heuristic
    g_score = {spot: float('inf') for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float('inf') for row in grid for spot in row}
    f_score[start] = g_score[start] + h(start.get_pos(), end.get_pos())

    # reached
    open_set_hash = {start}

    while not open_set.empty():
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()

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
            return (True, length, came_from, came_to)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                came_to[current] = neighbor
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

    return (False, -1, came_from, came_to)


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

    random_leak = random.choice(list(white))
    # random_leak2 = random.choice(list(white - {random_leak}))
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


def Bot3(width, ROWS, square, ALPHA):
    grid = make_grid(ROWS, width)

    # Gets the number of open cells from the ship
    may_contain_leak, random_bot, random_leak = make_ship(
        lambda: draw(win, grid, ROWS, width), grid, ROWS, square=square)

    start = random_bot

    # Sets the probabilities to a default value
    probabilities = defaultdict(lambda: 1/len(may_contain_leak))
    for i in may_contain_leak:
        probabilities[i.get_pos()] = 1/len(may_contain_leak)

    # Updates the probabilities every time the bot enters cell
    def bot_enters_cell_probability_update(probability_matrix, bot_location):
        #this update is P(leak in cell i given leak in cell j) / P(leak in cell i given cell in j') for each j'
        for key in probability_matrix:
            #since the sum of the numerator and denominator is equal to 1, we can calculate denominator by doing
            # 1- P(leak in bot_location) or one minnus our previously believed probability
            denom = 1 - probability_matrix[bot_location]
            probability_matrix[key] = probability_matrix[key] / denom
        # key is position of cell j we want to calculate updated probability for
        probability_matrix[bot_location] = 0
        return probability_matrix

    # Updates all the probabilities every time a beep occurs
    def beep_probability_update(probability_matrix, bot_location):
        #after we enter a cell where a beep is heard, we know there is no leak in this cell
        # so we make the probability that the leak is in the current cell to 0
        
        probability_matrix[bot_location] = 0
        # after doing so we normalize the remaining probabilities to account for this update
        #denom represents the summation in the denominator from the probability update calculation
        #summation of denominator in probability function is P(beep in cell i given leak in cell j') for each j' in open cells
        denom = sum(
            probability_matrix[key2] *
            (E**((-1 * ALPHA) * (dists[(bot_location, key2)] - 1)))
            for key2 in probability_matrix
            if key2 != bot_location
        )
        #Probability calculation uses this sum, to normalize each of the remaining cells' probability to contain the leak, as discussed in class and announcements/discussion posts
        #probability function numerator uses P(beep in cell i given leak in cell j)
        for key in probability_matrix:

            if denom != 0 and not math.isinf(denom):
                probability_matrix[key] = (
                    probability_matrix[key] *
                    (E**((-1 * ALPHA) * (dists[(bot_location, key)] - 1)))
                ) / denom

        return probability_matrix

    # Updates all the probabilitites every time a beep does not occur
    def no_beep_probability_update(probability_matrix, bot_location):
        #probability update for no_beep_probability_update follows beep_probability_update 
        probability_matrix[bot_location] = 0
        #the difference lies in the P(no beep in i given leak in j) probability which is 
        #1 - P(beep in i given leak in j)
        denom = sum(
            probability_matrix[key2] *
            (1 - (E**((-1 * ALPHA) *
                     (dists[(bot_location, key2)] - 1))))
            for key2 in probability_matrix
            if key2 != bot_location
        )
        for key in probability_matrix:

            if denom != 0 and not math.isinf(denom):

                probability_matrix[key] = (
                    probability_matrix[key] * (1 - (E**((-1 * ALPHA)
                                               * (dists[(bot_location, key)] - 1))))
                ) / denom

                #print(probability_matrix[key])
        return probability_matrix

    # Returns the row and col of the cell with the highest probability
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

    # Calculates all the distances in advance and stores them in a hashmap
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
                        og_nei.get_pos(), nei.get_pos())]

                    queue.append(nei)

    while run:
        for row in grid:
            for spot in row:
                spot.update_neighbors(grid)
                spot.update_unres_neighbors(grid)
        if time:
            next_location = None
            # Runs the code until the start location is the same as the end
            while (start.get_pos() != random_leak.get_pos()):
                # Gets the new probabilities
                probabilities = bot_enters_cell_probability_update(
                    probabilities, start.get_pos())

                # Checks if bot is currently moving or not
                sense_again = all(not i.is_path() for i in start.neighbors)

                # Runs if the bot is currently not moving
                if sense_again:

                    # The probability of beep occuring
                    total_actions += 1
                    beep = random.random() <= (
                        E**((-1*ALPHA)*(dists[start.get_pos(), random_leak.get_pos()] - 1)))
                    # Depending  on if beep occurs or not the probabilities get updated
                    if beep:
                        probabilities = beep_probability_update(
                            probabilities, start.get_pos())
                    else:
                        probabilities = no_beep_probability_update(
                            probabilities, start.get_pos())

                    # Gets the next location with the highest prob and goes there
                    next_location = get_location_of_max_probability(
                        probabilities)

                    a, temp, came_from, came_to = algorithm(lambda: draw(win, grid, ROWS, width),
                                                            grid, start, next_location)

                # get path from bot location to the next location found
                #print(sum(probabilities.values()))
                random_leak.make_color(BROWN)

                # Edge case
                for i in start.neighbors:
                    # Checks if the leak is in the vicinity of the path of the movement
                    if i.get_pos() == random_leak.get_pos():
                        browncount = 0
                        for j in i.neighbors:
                            if j.is_path() or j.get_pos() == start.get_pos() or j.get_pos() == next_location.get_pos():
                                browncount += 1
                        if browncount == 2:
                            total_actions += 1
                            time = False
                            run = False
                    # If it is in the path, just move forward, don't end
                    if i.is_path() or i.get_pos() == next_location.get_pos():

                        i.make_start()
                        start.reset()
                        start = i
                        # print(start.get_pos())
                        if start.get_pos() == random_leak.get_pos():
                            time = False
                            run = False
                        else:
                            # Updates probabilities
                            probabilities = bot_enters_cell_probability_update(
                                probabilities, start.get_pos())
                            total_actions += 1
            time = False
            run = False

    return total_actions


# Runs the bot in parallel with multiprocessing
def run_bot3(alpha):
    WIDTH = 800
    # WIN = pygame.display.set_mode((WIDTH, WIDTH))
    # pygame.display.set_caption("Leak Finding Algorithm")
    ROWS = 50
    total_actions = 0
    count = 0
    for i in range(1500):
        try:
            total_actions += Bot3(WIDTH, ROWS, 3, alpha)
        except Exception as e:
            print(f"Error in execution for alpha={alpha}: {e}", flush=True)
            traceback.print_exc()
            print("SKLDJFHLSDFH", flush=True)
            return 0
            count -= 1
        count += 1
        print(count, flush=True)
    # pygame.quit()
    print("FINISHED", alpha, flush=True)
    gc.collect()
    return total_actions/(count)


def main():
    success = defaultdict(int)
    alphas = [i / 1000 for i in range(1, 101)]
    with ProcessPoolExecutor(max_workers=10) as executor:

        futures = {executor.submit(run_bot3, alpha): alpha for alpha in alphas}
        count = 0
        for future in as_completed(futures):
            alpha = futures[future]
            result = future.result()
            success[alpha] += result
            count += 1
            print(count, alpha)

    print(success)

    alphas, total_actions = zip(*sorted(success.items()))

    # Convert to NumPy arrays
    alphas = np.array(alphas)
    total_actions = np.array(total_actions)

    # Create the plot
    plt.scatter(alphas, total_actions, marker='o', linestyle='-', color='b')
    plt.title('Alpha vs Total Actions')
    plt.xlabel('Alpha')
    plt.ylabel('Total Actions')
    plt.grid(False)
    plt.savefig('bot_3.png')


if __name__ == "__main__":
    main()
