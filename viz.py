import numpy as np
import socket
import sys

def say_hi(over):
    if over:
        try:
            sock.sendall('0/1')
        finally:
            sock.close()
    else:
        sock.sendall('0/1')

def init():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 31415)
    sock.connect(server_address)

def show(avg, ver, over):
    if over:
        try:
        finally:
            sock.close()
    else:
        sock.sendall('0')
        for i in range():
            sock.sendall('0')
            sock.sendall(convMat(avg[:,i]))
