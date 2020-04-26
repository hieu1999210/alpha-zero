import torch
import time
import torch.nn as nn
import os
from tqdm import tqdm
from torch.multiprocessing import Pool, Process, set_start_method, Queue, Value, Event
from torchvision.models import resnet18
from modelling import Model
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     print("runtime error")
#     pass

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(100,100),
        #     nn.Linear(100,100),
        #     nn.Linear(100,100),
        #     nn.Linear(100,100),
        #     nn.Linear(100,100),
        #     # nn.Linear(100,100),
        #     # nn.Linear(100,100),
        #     # nn.Linear(100,100),
        #     nn.Linear(100,1),
        # )
        self.backbone = resnet18(num_classes=1)
        out_channels = self.backbone.conv1.out_channels
        self.backbone.conv1 = nn.Conv2d(
            1, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.backbone(x)

class MyProcess(Process):
    "only communicate by a global count"
    def __init__(
        self, pid, model, output_queue, num_completed, 
        inp_tensor, finished_event, total_runs
    ):
        super(MyProcess, self).__init__()
        self.model = model
        self.output_queue = output_queue
        self.num_completed = num_completed
        self.finished_event = finished_event
        self.total_runs = total_runs
        self.inp_tensor = inp_tensor
        self.pid___ = pid
    def run(self):
        count = 0
        assert not self.model.training, "model not in eval mode"
        print(f"process {self.pid___} started")
        while self.num_completed.value < self.total_runs:
            # print(f"process {self.pid___} run {count+1}-th step")
            
            # for _ in range(100):
            self.inp_tensor += 1
            out = self.model(self.inp_tensor)
            self.output_queue.put(out.detach().cpu().numpy())
            with self.num_completed.get_lock():
                self.num_completed.value += 1
            count += 1
        
        print(f"process {self.pid___} finished, runned {count} times")
        self.finished_event.set()

class Runner:
    def __init__(self, model, device, num_worker=5, total_runs=100):
        self.num_completed = Value('i', 0)
        self.output_buffer = Queue()
        self.model = model
        self.input_tensor = [
            torch.zeros((4,1,6,6)).pin_memory().to(device).half().share_memory_() for _ in range(num_worker)
        ]
        self.process_done = [Event() for _ in range(num_worker)]
        self.processes = [MyProcess( i,
            model, self.output_buffer, self.num_completed, 
            self.input_tensor[i], self.process_done[i], total_runs
        ) for i in range(num_worker)]
        print("done init")
    def run(self):
        start = time.perf_counter()
        try:
            for p in self.processes:
                p.start()
            
            for event in self.process_done:
                event.wait()
            a = self.output_buffer.qsize()
            for i in tqdm(range(a)):
                self.output_buffer.get()
            print(f"got {a} results")
        except Exception as e:
            print(e)
            print("some error")
        finally:
            for p in self.processes:
                p.join()
            print(f"all processes joined")
        print("Elappsed time: ", time.perf_counter() - start)
if __name__ == "__main__":
    # with torch.no_grad():
    # device = "cpu"
    device = "cuda"
    set_start_method('spawn', force=True)
    # model = Mymodel()
    model = Model(None)
    for param in model.parameters():
        param.requires_grad = False
    model.half()
    model = model.to(device).share_memory()
    model.eval()
    runner = Runner(model, device, 1, total_runs=64)
    runner.run()
    