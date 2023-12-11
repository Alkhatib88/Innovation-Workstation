import asyncio
import socket
import struct
import enum

# Define an enum for different types of data
class DataType(enum.Enum):
    TEXT = 1
    IMAGE = 2
    FILE = 3
    PACKAGE = 4

class TCPServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = []
        self.data = None
        self.header = "utf-8"

    def settings(self, host, port, max_clients):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.server.bind((self.host, self.port))

    async def listen(self):
        self.server.listen()
        print(f"Server is listening on {self.host}:{self.port}...")

    async def accept_clients(self):
        while True:
            client, address = await self._accept()
            if len(self.clients) >= self.max_clients:
                client.close()
                print(f"Client {address} tried to connect, but the server is full")
                continue
            print(f"Client connected from {address}")
            self.clients.append(client)
            asyncio.ensure_future(self.handle_client(client))

    async def handle_client(self, client):
        while True:
            data_type = await self._recv(client, 1)  # Receive 1 byte for data type
            data_len = await self._recv_len(client)
            if not data_len:
                self.clients.remove(client)
                client.close()
                break
            data = await self._recv(client, data_len)
            if not data:
                self.clients.remove(client)
                client.close()
                break
            self.data = (data_type, data)

    async def send(self, client, data_type, data):
        # Send data type
        client.send(data_type.value.to_bytes(1, byteorder="big"))
        if isinstance(data, str):
            data = data.encode(self.header)
        data_len = struct.pack("!i", len(data))
        client.send(data_len)
        client.send(data)

    async def broadcast(self, data_type, data):
        for client in self.clients:
            await self.send(client, data_type, data)

    def kick(self, client):
        self.clients.remove(client)
        client.close()

    async def start(self):
        self.server.bind((self.host, self.port))
        await asyncio.gather(self.listen(), self.accept_clients())

    async def _accept(self):
        return await asyncio.get_running_loop().sock_accept(self.server)

    async def _recv(self, client, data_len):
        return await asyncio.get_running_loop().sock_recv(client, data_len)

    async def _recv_len(self, client):
        data_len = await asyncio.get_running_loop().sock_recv(client, 4)
        return struct.unpack("!i", data_len)[0]

    def get_data(self):
        data = self.data
        self.data = None
        return data

    def get_clients(self):
        return self.clients

server = TCPServer()
