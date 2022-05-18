def read_ram(env, start, end=None):
    """
    Reads the bytes at the given addresses.
    :param env: The ram environment (needs to contain a 'ram' variable or wrap an environment that does)
    :param start: From which byte we want to read
    :param end: Until which byte we want to read (included)
    :return: An array of decimal values for each byte read
    """

    return [env.ram[a] for a in range(start, start + 1 if end is None else end, 1)]