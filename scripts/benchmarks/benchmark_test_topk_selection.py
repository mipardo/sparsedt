import time
import torch


def get_threshold(tensor, k):
    sorted_tensor_values, _ = torch.sort(torch.abs(tensor).flatten())
    threshold = sorted_tensor_values[max(-k, -len(sorted_tensor_values))]
    return threshold


def topth_selection_custom(tensor, threshold):
    flattened_tensor = tensor.flatten()
    indexes = torch.where(flattened_tensor.abs() >= threshold)[0]
    topk = flattened_tensor[indexes]
    return topk, indexes


def topk_selection_custom(tensor, k):    
    threshold = get_threshold(tensor, k)
    return topth_selection_custom(tensor, threshold)


def topk_selection_pytorch(tensor, k):
    flattened_tensor = tensor.flatten()
    _, indexes = torch.topk(flattened_tensor.abs(), k, dim=0, sorted=False) # sorted=True sorts by magnitude, not indices
    sorted_indexes = indexes.sort(descending=False).values
    topk = flattened_tensor[sorted_indexes]
    return topk, sorted_indexes


def topk_selection_original(tensor, k):
    flattened_tensor = tensor.flatten()
    indexes = torch.abs(flattened_tensor).argsort()[-k:]
    sorted_indexes = indexes.sort().values
    return flattened_tensor[sorted_indexes], sorted_indexes


def run_functions(topk_methods, ks, tensors):
    for k in ks:
        print(f"Results with k = {k}:")
        for tensor in tensors:
            if tensor.numel() < k: 
                continue
            outputs = []
            print(f"  and tensor = {tensor.shape}:")
            # Run and measure method
            for topk in topk_methods:
                if topk.__name__ == "topth_selection_custom":
                    threshold = get_threshold(tensor, k)
                    start_time = time.time()
                    outputs.append(topth_selection_custom(tensor, threshold)) 
                    running_time = (time.time() - start_time) * 1000
                else:
                    start_time = time.time()
                    outputs.append(topk(tensor, k)) 
                    running_time = (time.time() - start_time) * 1000
                print(f"    - Method {topk.__name__} took {round(running_time, 4)} (ms)")
            # Assert that outputs are equal
            for i in range(1, len(topk_methods)):
                if not torch.equal(outputs[i][0], outputs[i - 1][0]) or not torch.equal(outputs[i][1], outputs[i - 1][1]):
                    raise AssertionError(f"Outputs are not equal:\n - " + "\n - ".join(str(output) for output in outputs) + "\n")
    


if __name__ == "__main__":
    
    torch.manual_seed(0)
    ks = [1, 2, 10, 100, 1000, 10_000, 100_000, 1_000_000]
    tensors = [torch.normal(mean=0.0, std=0.3, size=(128, 4,   32,  4)),
               torch.normal(mean=0.0, std=0.3, size=(128, 32,  128, 32)),
               torch.normal(mean=0.0, std=0.3, size=(128, 64,  128, 64))]
    topk_methods = [topk_selection_custom, topk_selection_pytorch, topk_selection_original, topth_selection_custom]
    run_functions(topk_methods, ks, tensors)
            
        