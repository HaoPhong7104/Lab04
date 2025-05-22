import time
import json
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
from torchvision.datasets import MNIST
from torchvision import transforms

parser = argparse.ArgumentParser(description='Streams MNIST data to a Spark Streaming Context')
parser.add_argument('--folder', '-f', help='Data folder', required=True, type=str)
parser.add_argument('--batch-size', '-b', help='Batch size', required=True, type=int)
parser.add_argument('--endless', '-e', help='Enable endless stream', action='store_true')
parser.add_argument('--split','-s', help="train or test split", required=False, type=str, default='train')
parser.add_argument('--sleep','-t', help="streaming interval (seconds)", required=False, type=float, default=3.0)

TCP_IP = "localhost"
TCP_PORT = 6100

class DatasetStreamer:
    def __init__(self, folder, batch_size, sleep_time, split):
        self.folder = folder
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.split = split.lower() == 'train'
        self.transform = transforms.ToTensor()
        self.dataset = MNIST(root=self.folder, train=self.split, download=True)

    def connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for Spark to connect at {TCP_IP}:{TCP_PORT}...")
        conn, addr = s.accept()
        print(f"Connected to Spark from {addr}")
        return conn

    def stream(self, conn, endless=False):
        print(f"Streaming MNIST {'endlessly' if endless else ''}...")
        while True:
            pbar = tqdm(total=len(self.dataset) // self.batch_size)
            for i in range(0, len(self.dataset), self.batch_size):
                batch_data = {}
                for j in range(self.batch_size):
                    if i + j >= len(self.dataset):
                        break
                    image, label = self.dataset[i + j]
                    image = image.view(-1).tolist()
                    sample = {f'feature-{k}': v for k, v in enumerate(image)}
                    sample['label'] = label
                    batch_data[j] = sample

                payload = (json.dumps(batch_data) + "\n").encode()
                try:
                    conn.send(payload)
                except Exception as e:
                    print(f"Connection error: {e}")
                    return

                time.sleep(self.sleep_time)
                pbar.update(1)
            pbar.close()

            if not endless:
                break

if __name__ == "__main__":
    args = parser.parse_args()

    streamer = DatasetStreamer(
        folder=args.folder,
        batch_size=args.batch_size,
        sleep_time=args.sleep,
        split=args.split
    )

    conn = streamer.connect()
    streamer.stream(conn, endless=args.endless)
    conn.close()
