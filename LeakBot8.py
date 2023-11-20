from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
from collections import defaultdict, deque
import gc
import math
import traceback
# import pygame
import random
from queue import PriorityQueue
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 800
# pygame.init()
# WIN = pygame.display.set_mode((WIDTH, WIDTH))
# pygame.display.set_caption("Leak Finding Algorithm")

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

    random_leak = random.choice(list(white - {random_bot}))
    random_leak2 = random.choice(list(white - {random_bot} - {random_leak}))
    # random_leak2 = random.choice(list(white - {random_leak}))
    random_leak.make_end()

    return (white, random_bot, random_leak, random_leak2)


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


def Bot8(width, ROWS, square, ALPHA):

    # assert square >= 3
    grid = make_grid(ROWS, width)

    may_contain_leak, random_bot, random_leak, random_leak2 = make_ship(
        lambda: draw(win, grid, ROWS, width), grid, ROWS, square=square)

    # dists = create_dist_matrix(may_contain_leak)
    # for key, value in dists:
    #     print(type(key), type(value))

    # may_contain_leak = may_contain_leak - {random_bot}

    start = random_bot

    # key is a position
    # val is a probability (float)
    probabilities = {}
    for i in may_contain_leak:
        for j in may_contain_leak:
            if j != i:
                probabilities[(i.get_pos(), j.get_pos())] = 1 / \
                    (len(may_contain_leak)**2)
    # print(len(probabilities))
    # probabilities[start.get_pos()] = 0

# for each sense function, set the bot_location( current location of bot), probability of leak to 0, since we have already visited it

    def bot_enters_cell_probability_update(probability_matrix, bot_location):
        # print(len(probability_matrix))
        # print("thinkgin")
        for (key, key2) in probability_matrix:
            # key is position of cell j we want to calculate updated probability for
            # key 2 is position of every other cell j', used for summation stored in denom
            denom = 1

            if key != bot_location:
                denom -= (probability_matrix[(key, bot_location)] +
                          probability_matrix[(bot_location, key)])
            if key2 != bot_location:
                denom -= (probability_matrix[(key2, bot_location)] +
                          probability_matrix[(bot_location, key2)])
            # if denom != 0 and not math.isinf(denom):
            try:
                probability_matrix[(key, key2)] = probability_matrix[(
                    key, key2)] / denom
            except:
                return False, probability_matrix

        for cell in may_contain_leak:
            if cell != bot_location:
                probability_matrix[(cell.get_pos(), bot_location)] = 0
                probability_matrix[(bot_location, cell.get_pos())] = 0
        return True, probability_matrix

    def beep_probability_update(probability_matrix, bot_location):
        # print("beep doing")
        for cell in may_contain_leak:
            if cell != bot_location:
                probability_matrix[(cell.get_pos(), bot_location)] = 0
                probability_matrix[(bot_location, cell.get_pos())] = 0
        # key is j, key 2 is k, key 4 is j', key 3 is k' bot_location is i

        denom = sum(
            probability_matrix[(key4, key3)] *
            (1 - ((1 - E**((-1 * ALPHA) * (dists[(bot_location, key4)] - 1))) *
                  (1 - E**((-1 * ALPHA) * (dists[(bot_location, key3)] - 1)))))
            for (key4, key3) in probability_matrix
            if key3 != bot_location and key4 != bot_location and key3 != key4
        )
        for (key, key2) in probability_matrix:
            # print((key,key2))
            if denom != 0 and not math.isinf(denom) and key != bot_location and key2 != bot_location:
                probability_matrix[(key, key2)] = (
                    probability_matrix[(key, key2)] *
                    (1 - ((1 - E**((-1 * ALPHA) * (dists[(bot_location, key)] - 1))) *
                          (1 - E**((-1 * ALPHA) * (dists[(bot_location, key2)] - 1)))))
                ) / denom

        return probability_matrix

    def no_beep_probability_update(probability_matrix, bot_location):
        # print("Beep boop")
        for cell in may_contain_leak:
            if cell != bot_location:
                probability_matrix[(cell.get_pos(), bot_location)] = 0
                probability_matrix[(bot_location, cell.get_pos())] = 0
        # key is j, key 2 is k, key 4 is j', key 3 is k' bot_location is i
        denom = sum(
            probability_matrix[(key4, key3)] *
            ((1 - E**((-1 * ALPHA) * (dists[(bot_location, key4)] - 1))) *
             (1 - E**((-1 * ALPHA) * (dists[(bot_location, key3)] - 1))))
            for (key4, key3) in probability_matrix
            if key3 != bot_location and key4 != bot_location and key3 != key4
        )
        for (key, key2) in probability_matrix:
            # print((key,key2))
            # print("h")
            if denom != 0 and not math.isinf(denom) and key != bot_location and key2 != bot_location:
                probability_matrix[(key, key2)] = (
                    probability_matrix[(key, key2)] *
                    ((1 - E**((-1 * ALPHA) * (dists[(bot_location, key)] - 1))) *
                     (1 - E**((-1 * ALPHA) * (dists[(bot_location, key2)] - 1))))
                ) / denom
        return probability_matrix

    def get_location_of_max_probability(probability_matrix):
        max_probability = 0
        max_location = None

        for key, probability in probability_matrix.items():
            if probability > max_probability:
                max_probability = probability
                # Assuming key is of format (spot.get_pos(), spot2.get_pos())
                max_location = key[0]

        # Assuming grid is the 2D list of Spot objects
        row, col = max_location
        max_spot = grid[row][col]

        return max_spot

    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)

    run = True
    time = True
    total_actions = 0

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
        # clock.tick(FPS)
        for row in grid:
            for spot in row:
                spot.update_neighbors(grid)
                spot.update_unres_neighbors(grid)
        # draw(win, grid, ROWS, width)#Here
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         run = False
        make_brown = True
        make_brown2 = True

        if time:
            next_location = None
            # print(start.get_pos())
            # pseudocode: while bot_location != leak_location:
            counter = 0

            while (counter < 2):
                # for _ in range(100):
                # print(sum(probabilities.values()))

                # Find next spot to explore
                sense_again = all(not i.is_path() for i in start.neighbors)

                # if not next_location or start.get_pos() == next_location.get_pos():
                if sense_again:

                    # print("reached")
                    total_actions += 1

                    beep = (1 - ((1 - E**((-1 * ALPHA) * (dists[(start.get_pos(), random_leak.get_pos())] - 1))) *
                                 (1 - E**((-1 * ALPHA) * (dists[(start.get_pos(), random_leak2.get_pos())] - 1)))))
                    if beep:
                        # print("BEEPING")
                        probabilities = beep_probability_update(
                            probabilities, start.get_pos())
                    else:
                        # print("NOT BEEPING")
                        probabilities = no_beep_probability_update(
                            probabilities, start.get_pos())
                    next_location = get_location_of_max_probability(
                        probabilities)
                    # print(next_location.get_pos())
                    # print(probabilities)

                    a, temp, came_from, came_to = algorithm(lambda: draw(win, grid, ROWS, width),
                                                            grid, start, next_location)
                    # total_actions += temp
                    # print(len(came_from))

                # get path from bot location to the next location found

                # while i!= next_location:
                # i = came_to[start]
                if make_brown:
                    random_leak.make_color(BROWN)
                if make_brown2:
                    random_leak2.make_color(BROWN)

                # print(i.get_pos(), start.get_pos())

                for i in start.neighbors:
                    if i.is_path() or i.get_pos() == next_location.get_pos():

                        # MAKE NEXT LOCATION BROWN CASE CODE HERE!!!!!
                        if i.get_pos() == random_leak.get_pos() or i.get_pos() == random_leak2.get_pos():

                            if i.get_pos() == random_leak.get_pos():
                                for cell in may_contain_leak:
                                    if cell != random_leak:
                                        probabilities[(
                                            cell.get_pos(), random_leak.get_pos())] = 0
                                        probabilities[(
                                            random_leak.get_pos(), cell.get_pos())] = 0

                                may_contain_leak = may_contain_leak - \
                                    {random_leak}
                                for k in may_contain_leak:
                                    for l in may_contain_leak:
                                        probabilities[(k.get_pos(), l.get_pos(
                                        ))] = 1/(len(may_contain_leak)**2)
                                random_leak = random_leak2
                                make_brown = False

                                # print("Leak 1")
                                counter += 1
                                if counter == 2:
                                    time = False
                                    run = False
                            elif i.get_pos() == random_leak2.get_pos():
                                for cell in may_contain_leak:
                                    if cell != random_leak2:
                                        probabilities[(
                                            cell.get_pos(), random_leak2.get_pos())] = 0
                                        probabilities[(
                                            random_leak2.get_pos(), cell.get_pos())] = 0

                                may_contain_leak = may_contain_leak - \
                                    {random_leak2}
                                for k in may_contain_leak:
                                    for l in may_contain_leak:
                                        probabilities[(k.get_pos(), l.get_pos(
                                        ))] = 1/(len(may_contain_leak)**2)
                                random_leak2 = random_leak
                                make_brown2 = False

                                # print("Leak 2")
                                counter += 1
                                if counter == 2:
                                    time = False
                                    run = False
                        i.make_start()
                        # may_contain_leak = may_contain_leak - {i}
                        start.reset()
                        start = i
                        # print(start.get_pos())
                        # print("reached 4")
                        works, probabilities = bot_enters_cell_probability_update(
                            probabilities, start.get_pos())
                        if not works:
                            time = False
                            run = False
                        # print("reached")
                        # probabilities[start.get_pos()] = 0
                        total_actions += 1
                    elif i.get_pos() == random_leak.get_pos() or i.get_pos() == random_leak2.get_pos():
                        # print("reached 3")
                        browncount = 0
                        for j in i.neighbors:
                            # print(j.get_pos(), j.get_color())
                            if j.is_path() or j.get_pos() == start.get_pos() or j.get_pos() == next_location.get_pos() or j.get_color() == BROWN:
                                browncount += 1
                        # print(browncount)
                        # print(next_location.get_pos())
                        if browncount >= 2:
                            if i.get_pos() == random_leak.get_pos():
                                for cell in may_contain_leak:
                                    if cell != random_leak:
                                        # print("reached 2")
                                        probabilities[(
                                            cell.get_pos(), random_leak.get_pos())] = 0
                                        probabilities[(
                                            random_leak.get_pos(), cell.get_pos())] = 0

                                may_contain_leak = may_contain_leak - \
                                    {random_leak}
                                for k in may_contain_leak:
                                    for l in may_contain_leak:
                                        probabilities[(k.get_pos(), l.get_pos(
                                        ))] = 1/(len(may_contain_leak)**2)
                                random_leak = random_leak2
                                make_brown = False

                                # print("Leak 1")
                                counter += 1
                                if counter == 2:
                                    time = False
                                    run = False
                            elif i.get_pos() == random_leak2.get_pos():
                                for cell in may_contain_leak:
                                    if cell != random_leak2:
                                        probabilities[(
                                            cell.get_pos(), random_leak2.get_pos())] = 0
                                        probabilities[(
                                            random_leak2.get_pos(), cell.get_pos())] = 0

                                may_contain_leak = may_contain_leak - \
                                    {random_leak2}
                                for k in may_contain_leak:
                                    for l in may_contain_leak:
                                        probabilities[(k.get_pos(), l.get_pos(
                                        ))] = 1/(len(may_contain_leak)**2)
                                random_leak2 = random_leak
                                make_brown2 = False

                                # print("Leak 2")
                                counter += 1
                                if counter == 2:
                                    time = False
                                    run = False
                            return total_actions

                            # return total_actions

                # pygame.time.delay(1000)
                # draw(win, grid, ROWS, width)
            time = False
            run = False

    # pygame.quit()
    return total_actions


# def main(win, width):
#     ROWS = 30
#     # make them return FAILED OR SUCCEEDED, ALSO PASS IN Q
#     # actions = Bot3(win, width,  ROWS, 3, 0.5)
#     # print(actions)
#     # actions = Bot3(win, width,  ROWS, 3, 0.5)
#     # print(actions)
#     success = defaultdict(int)
#     count_set = 0
#     # count = 0
#     for i in range(1, 2):
#         count_set += 1
#         print(count_set)
#         for _ in range(1):
#             # count += 1
#             # print(count)
#             success[i/10] += Bot3(win, width,  ROWS, 3, i/10)
#     print(success)


# Your existing main method

def run_bot3(alpha):
    WIDTH = 800
    # pygame.init()
    # WIN = pygame.display.set_mode((WIDTH, WIDTH))
    # pygame.display.set_caption("Leak Finding Algorithm")
    ROWS = 8
    total_actions = 0
    count = 0
    for i in range(1):
        try:
            # print("hi")
            total_actions += Bot8(WIDTH, ROWS, 3, alpha)
        except Exception as e:
            print(f"Error in execution for alpha={alpha}: {e}", flush=True)
            traceback.print_exc()
            print("SKLDJFHLSDFH", flush=True)

            count -= 1
        count += 1
        print(count, flush=True)
    # pygame.quit()
    print("FINISHED", alpha, flush=True)
    gc.collect()
    return total_actions/(count)


def main():
    success = defaultdict(int)
    alphas = [i / 200 for i in range(1, 21)]
    with ProcessPoolExecutor(max_workers=3) as executor:

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
    plt.savefig('bot_8.png')


if __name__ == "__main__":
    main()
