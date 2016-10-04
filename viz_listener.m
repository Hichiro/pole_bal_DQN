clear all;
close all;
clc;

pkg load sockets

sock = socket();
bind(sock, 31415);
listen(sock, 0);
client = accept(sock);
len = 8;
[data, count] = recv(client, len);
data
count
fflush(1);
