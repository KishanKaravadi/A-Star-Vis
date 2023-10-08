import pygame
import math
import random
from queue import PriorityQueue

WIDTH = 800
Q = 1
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
        self.num_fire_nei = 0

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == TURQUOISE

    def is_end(self):
        return self.color == GREEN_YELLOW

    def is_fire(self) -> bool:
        return self.color == ORANGE

    def reset(self):
        self.color = WHITE

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

    def make_end(self):
        self.color = GREEN_YELLOW

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        #need to check the is_fire node with the set of fire cells, we can change is_fire to do this
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


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def algorithm(draw, grid, start, end, orange, white, Q):
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
        pygame.time.delay(1000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # update board here using a method that updates entire board based on fire
        # must be done here so that current doesn't equal a fire cell
        #print(orange)
        if orange:
            cell = orange.pop()
            # if cell in open_set.queue:
            #     open_set.queue.remove(cell)
            for orange_nei in cell.unres_neighbors:
                #print(orange_nei.num_fire_nei)
                if orange_nei.get_color()!=BLACK and random.random() <= 1:
                    if orange_nei.is_end() or orange_nei.is_start() or orange_nei.get_color==PURPLE: 
                        #orange_nei.make_color==BROWN
                        #draw()
                        #print("HURIBUIBFBFO")
                        return False 
                    orange_nei.make_color(ORANGE)
                    for orange_neinei in orange_nei.neighbors:
                        orange_neinei.num_fire_nei+=1
                    orange.add(orange_nei)
                    if orange_nei in white:
                        white.remove(orange_nei)
                    for tup in open_set.queue:
                        #print(tup[2].get_pos)
                        if orange_nei in tup:
                            print("removed")
                            print((tup[2].row,tup[2].col))
                            open_set.queue.remove(tup)
                            open_set_hash.remove(orange_nei)
        #fulfills set of fire nodes (fire nodes should store their neighbors' locations for #3)
        
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            start.make_start()
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash and not neighbor.is_fire():
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False


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


# ???


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
    orange = set()

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
        #pygame.time.delay(500)

    # Check for deadends
    for cell in white:
        # print((cell.row, cell.col, cell.num_white_nei))
        if cell.num_white_nei == 1:
            brown.add(cell)
            cell.make_color(BROWN)

        draw()
        #pygame.time.delay(500)

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
        #pygame.time.delay(2000)
    draw()
    # pygame.time.delay(1000)

    # Left over brown made into white
    while brown:
        leftover = brown.pop()
        leftover.make_color(WHITE)

    draw()
    #pygame.time.delay(1000)
    while red:
        temp = red.pop()
        temp.make_color(BLACK)

    #pygame.time.delay(1000)
    draw()

    random_bot = random.choice(list(white))
    random_bot.make_start()

    random_button = random.choice(list(white - {random_bot}))
    random_button.make_end()

    random_fire = random.choice(list(white - {random_bot, random_button}))
    random_fire.make_color(ORANGE)
    for firecell in random_fire.neighbors:
        firecell.num_fire_nei+=1
    #need to add to set of fire nodes
    orange.add(random_fire)
    white.remove(random_fire)

    draw()

    return (random_bot, random_button, random_fire, orange, white)


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


# why are x and y switched
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap
    return row, col


def main(win, width):
    ROWS = 10
    grid = make_grid(ROWS, width)
    random_bot, random_button, random_fire, orange, white = make_ship(
        lambda: draw(win, grid, ROWS, width), grid, ROWS)

    start = random_bot
    end = random_button
    #print((random_bot.get_pos()[0],random_bot.get_pos()[1]))
    print((random_fire.get_pos()[0],random_fire.get_pos()[1]))
    for nei in random_fire.neighbors:
        print((nei.get_pos()[0],nei.get_pos()[1]))

    run = True
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
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right Mouse Button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
            elif pygame.mouse.get_pressed()[1]:  # Middle Mouse Button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.make_color(GREEN)
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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("True") if algorithm(lambda: draw(win, grid, ROWS, width),
                              grid, start, end, orange, white, Q) else print("False")

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
