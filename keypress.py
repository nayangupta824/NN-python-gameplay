from subprocess import Popen, PIPE
from random import *
import time

w_down = '''keydown w
'''
w_up = '''keyup w
'''
s_down = '''keydown s
'''
s_up = '''keyup s
'''
a_down = '''keydown a
'''
a_up = '''keyup a
'''
d_down = '''keydown d
'''
d_up = '''keyup d
'''

def keypress(sequence):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)


prev = 0
while True:
	keypress(w_down)
	if prev == 1:
		keypress(s_up)
	elif prev == 2:
		keypress(d_up)
	x = randint(1,2)
	print x
	if x == 1:
		keypress(s_down)
		prev = 1
	else:
		keypress(d_down)
		prev = 2
	time.sleep(2)

keypress(w_up)

