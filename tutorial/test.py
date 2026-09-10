mem_positions = [3, 2, 1, 0]
mem_pos = mem_positions[::-1]

for i in range(0, len(mem_pos), 2):
    mem_pos0 = mem_pos[i]
    mem_pos1 = mem_pos[i + 1]

    print(mem_pos0, mem_pos1)