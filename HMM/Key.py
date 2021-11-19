"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class Key(object):
    # We will consider a-z + SPACE, all in all 27 characters.
    NUMBER_OF_CHARS = 27

    # The START_END symbol is used to represent space, as well as beginning and end of line.
    START_END = NUMBER_OF_CHARS - 1

    # This array encodes the topology of the keyboard.
    neighbour = (
        ('q', 'w', 's', 'z'), # a
        ('v', 'g', 'h', 'n'), # b
        ('x', 'd', 'f', 'v'), # c
        ('x', 's', 'e', 'r', 'f', 'c'), # d
        ('w', 's', 'd', 'r'), # e
        ('d', 'r', 't', 'g', 'v', 'c'), # f
        ('f', 't', 'y', 'h', 'b', 'v'), # g
        ('g', 'y', 'u', 'j', 'n', 'b'), # h
        ('u', 'j', 'k', 'o'), # i
        ('h', 'u', 'i', 'k', 'm', 'n'), # j
        ('m', 'j', 'i', 'o', 'l'), # k
        ('k', 'o', 'p'), # l
        ('n', 'j', 'k'), # m
        ('b', 'h', 'j', 'm'), # n
        ('i', 'k', 'l', 'p'), # o
        ('o', 'l'), # p
        ('w', 'a'), # q
        ('e', 'd', 'f', 't'), # r
        ('a', 'w', 'e', 'd', 'x', 'z'), # s
        ('r', 'f', 'g', 'y'), # t
        ('y', 'h', 'j', 'i'), # u
        ('c', 'f', 'g', 'b'), # v
        ('q', 'a', 's', 'e'), # w
        ('z', 's', 'd', 'c'), # x
        ('t', 'g', 'h', 'u'), # y
        ('x', 's', 'a'), # z
        () # whitespace, represented by the START_END symbol
    )

    @staticmethod
    def char_to_index(c):
        if c >= 'a' and c <= 'z':
            return ord(c) - ord('a')
        else: # Return the START_END symbol for all symbols outside the a-z interval.
            return Key.START_END

    @staticmethod
    def index_to_char(i):
        if i == Key.START_END:
            return ' '

        if i < Key.NUMBER_OF_CHARS - 1:
            return chr(i + ord('a'))

        # This shouldn't happen
        return 0

    @staticmethod
    def whitespace(c):
        return c.isspace()
