
def one_hot(hot_idx, size):
    v = [0 for _ in range(size)]
    v[hot_idx] = 1
    return v