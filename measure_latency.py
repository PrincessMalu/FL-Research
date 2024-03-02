import torch
import time
import torchvision.models as models
import numpy as np
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import sys

batch_sizes = [1,2,4,8,16]
latencies = []

for size in batch_sizes:
    model = models.mobilenet_v3_small(pretrained=True)
    
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(size, 3,224,224, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE in (ms)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    latencies.append(mean_syn)
    time.sleep(10)
    print("Batch Size: ", size, "Avg Latency: ", mean_syn)

# print(latencies)

#plotting the graph of latencies
# plt.ion()
# plt.plot(batch_sizes, latencies, 'r*')
# plt.show