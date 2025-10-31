import torch 
import warnings
import numpy as np
from mpi4py import MPI
from torch.optim import Optimizer
from utils.sparse_np import SparseTensorCOO


class OkTopk(Optimizer):
    def __init__(self, model_params, lr, momentum, weight_decay, density, min_k_layer, tau, tau_prime):
        super().__init__(
            model_params, 
            defaults={
                "lr": lr, 
                "momentum" :momentum, 
                "weight_decay": weight_decay, 
                "density":density, 
                "min_k_layer": min_k_layer,
                "tau": tau, 
                "tau_prime":tau_prime
            })
        
        self.iterations = {}
        self.all_local_th = {}
        self.all_global_th = {}
        self.all_residuals = {}
        self.all_boundaries = {}
        self.info_messages = set()
        self.comm = MPI.COMM_WORLD

        for group_id, group in enumerate(self.param_groups):
            layers = len(group["params"])
            self.iterations[group_id] = 0
            self.all_local_th[group_id] = {layer: None for layer in range(layers)}
            self.all_global_th[group_id] = {layer: None for layer in range(layers)}
            self.all_residuals[group_id] = {layer: None for layer in range(layers)}
            self.all_boundaries[group_id] = {layer: None for layer in range(layers)}
        
        
    def step(self):
        for group_id, group in enumerate(self.param_groups):
            # For every layer
            for layer_id, layer_params in enumerate(group["params"]):
                if layer_params.grad is not None:
                    # Get layer grads
                    grads = layer_params.grad
                    
                    # Compute k from: layer_params * self.density
                    k = int(torch.numel(grads) * group["density"])
                    k = group["min_k_layer"] if k < group["min_k_layer"] else k
                    
                    # Initialize current layer-parameter values
                    self.local_th = self.all_local_th[group_id][layer_id]
                    self.global_th = self.all_global_th[group_id][layer_id]
                    self.boundaries = self.all_boundaries[group_id][layer_id]
                    if self.all_residuals[group_id][layer_id] is None:
                        self.all_residuals[group_id][layer_id] = torch.zeros_like(grads, dtype=grads.dtype)
                    
                    # Compute acc 
                    acc = self._compute_acc(self.all_residuals[group_id][layer_id], grads, group["lr"])
                    
                    # Main part of ok-topk: compute the values that contribute to the update and its indexes
                    coo_u, indexes = self._ok_sparse_allreduce(acc, self.iterations[group_id], k, group["tau"], group["tau_prime"])

                    # Update residuals
                    self.all_residuals[group_id][layer_id] = self._reset_residuals(acc, indexes)
                    
                    # Save for next updates thresholds and boundaries
                    self.all_local_th[group_id][layer_id] = self.local_th
                    self.all_global_th[group_id][layer_id] = self.global_th
                    self.all_boundaries[group_id][layer_id] = self.boundaries

                    # Perform the weights update
                    self._update_weights(coo_u, group["weight_decay"], group["momentum"], group["lr"], layer_params)
                    
            # Update iterations
            self.iterations[group_id] += 1
        
        
    def _compute_acc(self, residuals, grads, learning_rate):
        """
        Compute acc, where: acc = residuals + (learning_rate * dw)

        Parameters:
            residuals (torch.Tensor): A dense matrix with the current layer residuals
            grads (torch.Tensor): A dense matrix with the current layer gradients
            learning_rate (float): learning rate float value

        Returns:
            acc (torch.Tensor): 2D dense matrix with the updated residuals
        """
        
        with torch.no_grad():
            return residuals + (learning_rate * grads)
    
        
    def _reset_residuals(self, acc, indexes):
        """
        Update residuals: set zero value if it is in indexes, else acc value is set.
        
        Parameters:
            acc (torch.Tensor): Dense accumulators matrix
            indexes (torch.Tensor): Indexes to reset in 1D

        Returns:
            residuals (torch.Tensor): which is the same as acc with the values in indexes set to zero.
        """
        
        indexes_tensor = torch.from_numpy(indexes).long().to(acc.device)

        
        with torch.no_grad():
            if self.defaults["density"] == 1:
                return torch.zeros_like(acc)
            
            if len(indexes) > 0:
                acc[torch.unravel_index(indexes_tensor, acc.shape)] = 0    
            return acc
    
    
    def _update_weights(self, coo_u, weight_decay, momentum, learning_rate, params, method="sparse"):
        """
        Update weights: w -= (u / self.comm.size) 

        Parameters:

        Returns:
            (void): instead it directly applies the result to the weight layer attribute
        """
        
        self._show_message_only_once(f"In '_update_weights', the method that it is being used is '{method}'")
        
        if method == "dense":
            grads = coo_u.to_dense()
            # Apply weight decay
            if weight_decay != 0:
                grads = grads.add(params.data, alpha=weight_decay)
            # Apply momentum
            if momentum != 0:
                state = self.state[params]
                if "momentum_buffer" not in state:
                    buf = grads.clone()
                    state["momentum_buffer"] = buf
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grads)
                grads = buf
            # Update parameters
            params.data.add_(grads, alpha=-learning_rate)

        elif method == "sparse":
            w = params.data.view(-1)
            
            # Convertir los índices y valores de NumPy a tensores de PyTorch
            indexes_tensor = torch.from_numpy(coo_u.indexes).long().to(w.device)
            values_tensor = torch.from_numpy(coo_u.values).to(w.device, dtype=w.dtype)

            # 1) Weight decay desacoplado (AdamW-style): w <- (1 - lr*wd) * w
            if weight_decay != 0:
                w.mul_(1.0 - learning_rate * weight_decay)

            # 2) Momentum disperso (solo en índices tocados)
            if momentum != 0:
                state = self.state[params]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(w)
                m = state["momentum_buffer"]

                # m_i = momentum * m_i + g_i  (solo para i en idx)
                m_sel = m.gather(0, indexes_tensor)                  
                m_sel.mul_(momentum).add_(values_tensor)              
                m.index_copy_(0, indexes_tensor, m_sel)              
                g_to_apply = m_sel
            else:
                g_to_apply = values_tensor

            # 3) Actualización dispersa: w[idx] -= lr * g_to_apply (suma si hay duplicados)
            w.index_add_(0, indexes_tensor, g_to_apply, alpha=-learning_rate)
            
        else:
            raise NotImplementedError(f"Method {method} not yet implemented in _update_weights")
        
        
    def _ok_sparse_allreduce(self, acc, t, k, space_repartition_t, thresholds_re_evaluation_t):
        """
        Performs the Ok-Topk sparse allreduce operation. 
        This method executes the Ok-Topk sparse allreduce algorithm, which 
        optimizes communication by only exchanging the most significant 
        gradient values (top-k) across distributed processes. The method 
        periodically re-evaluates the thresholds and repartitions the 
        gradient space to maintain efficiency and accuracy.

        Parameters:
            acc (torch.Tensor): A dense gradient matrix accumulation values.
            t (int): Current iteration number.
            k (int): Number of top-k gradient values to select in the current layer.
            space_repartition_t (int): Interval of iterations for space repartitioning.
            thresholds_re_evaluation_t (int): Interval of iterations for threshold re-evaluation.

        Returns:
            out (tuple with two elements:):
                - coo_u (SparseTensorCOO): The updated gradient values in 2D sparse format.
                - indexes (tuple(np.array, np.array)): The indices of the top-k gradient values that were updated.
        """

        with torch.no_grad():
            if t % thresholds_re_evaluation_t == 0:
                self.local_th = self._th_re_evaluate(acc, k, input_format="dense")
            
            if t % space_repartition_t == 0:
                self.boundaries = self._space_repartition(acc, self.local_th)

            coo_reduced_region_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)
            
            if t % thresholds_re_evaluation_t == 0:
                coo_all_reduced_topk = self._allgather(coo_reduced_region_topk)
                self.global_th = self._th_re_evaluate(coo_all_reduced_topk, k, input_format="coo")

            coo_u, global_topk_indexes = self._balance_and_allgather(coo_reduced_region_topk, self.global_th)
            indexes = self._intersect_indexes(local_topk_indexes, global_topk_indexes)
            return coo_u, indexes
    
    
    def _th_re_evaluate(self, tensor, k, input_format="dense"):
        """
        Return the absolute gradient threshold for a given matrix.
        
        Parameters:
            tensor (torch.Tensor or SparseTensorCOO): A 2D gradient matrix, in np.array for 'dense' input_format or SparseTensorCOO for 'coo' input_format.
            k (int): Indicating the number of top gradient values to consider.
            input_format (string): Either 'dense' for a dense matrix or 'coo' for a sparse matrix in COO format.
            method (string, optional): The method to use for threshold selection. It can be 'numpy_sort' or 'numpy_partition'.
        
        Returns:
            threshold (float): The absolute gradient threshold based on the top k values.
        """

        if k <= 0:
            return 1.0
        
        if input_format == "coo" and tensor.nnz == 0:
            return 1.0

        if input_format == "dense":
            sorted_tensor, _ = torch.sort(torch.abs(tensor).flatten())
            threshold = sorted_tensor[max(-k, -len(sorted_tensor))]
            return threshold
            
        if input_format == "coo":
            sorted_data = np.sort(np.abs(tensor.values))
            threshold = sorted_data[max(-k, -len(sorted_data))]
            return threshold
    
    
    def _space_repartition(self, acc, local_th, balanced=False):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        Parameters:
            acc (torch.Tensor): Dense gradient values
            local_th (float): local process gradient threshold
            balanced (boolean, optional): if not balanced a static partition is performed, 
                                          if balanced a topk gradiend distribution is considered in the partition  

        Returns:
            boundaries (torch.Tensor): [end_p0, end_p1, end_p2, ...]
        """
        
        self._show_message_only_once(f"\nIn '_space_repartition', balanced = '{balanced}' is being used")
        
        if not balanced:
            boundaries = torch.zeros(self.comm.size, dtype=torch.int64)
            total_elements = torch.numel(acc)           
            block_size = total_elements // self.comm.size
            for i in range(0, self.comm.size - 1):
                boundaries[i] = block_size * (i + 1)
            boundaries[self.comm.size - 1] = total_elements
            return boundaries
        
        elif balanced:
            coo_topk = SparseTensorCOO.from_dense_top_selection(acc, local_th)
            
            proc, idx, topk_in_proc = 0, 0, 0
            total_dense_values = torch.numel(acc)
            topk_per_proc = coo_topk.nnz // self.comm.size
            boundaries = torch.zeros(self.comm.size, dtype=torch.int32)
            idx_counter = torch.zeros(total_dense_values, dtype=torch.int32)
            idx_counter.index_add_(0, coo_topk.indexes, torch.ones_like(coo_topk.indexes, dtype=torch.int))

            while proc < self.comm.size - 1:
                if idx < total_dense_values:
                    topk_in_proc += idx_counter[idx]
                    if topk_in_proc >= topk_per_proc:
                        boundaries[proc] = idx 
                        topk_in_proc = 0
                        proc += 1
                    idx += 1
                else:
                    boundaries[proc] = idx 
                    proc += 1
            boundaries[self.comm.size - 1] = total_dense_values
            global_boundaries = self.comm.allreduce(boundaries, op=MPI.SUM) // self.comm.size
            
            return global_boundaries
    
    
    def _split_and_reduce(self, acc, local_th, boundaries):
        """
        First main phase of ok_sparse_allreduce.  
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts the reduction locally. 

        Parameters:
            acc (torch.Tensor): The gradient matrix accumulation values in dense format.
            local_th (float): Local threshold for selecting top-k values.
            boundaries (torch.Tensor): Boundaries for partitioning the gradient space like [end_p0, end_p1, end_p2, ...]

        Returns:
            out (tuple with two elements:):
                - coo_reduced_region_topk (SparseTensorCOO): The reduced top-k gradient values in COO format.
                - local_topk_indexes (torch.Tensor): The indices of the top-k gradient values selected locally.
        """
        
        coo_topk = SparseTensorCOO.from_dense_top_selection(acc, local_th)
        coo_reduced_region_topk = self._reduce_topk(coo_topk, boundaries)
        return coo_reduced_region_topk, coo_topk.indexes
    
    
    def _allgather(self, local_data, input_format="SparseTensorCOO"):
        """
        Gathers data from all processes.
        
        Parameters:
            local_data (torch.Tensor or SparseTensorCOO): The local data to be gathered.
            input_format (str, optional): The format of the input data.
        Returns:
            gathered_data (torch.Tensor or SparseTensorCOO): The gathered global data in the specified format.
        """
        
        if self.comm.size == 1:
            return local_data
        
        if input_format == "SparseTensorCOO":
            gathered = self.comm.allgather((local_data.values, local_data.indexes))
            all_val = np.concatenate([t[0] for t in gathered])
            all_ind = np.concatenate([t[1] for t in gathered])
            return SparseTensorCOO(all_val, all_ind, local_data.shape, has_canonical_format=True)

        if input_format == "dense":
            warnings.warn("Try to avoid dense communications!")
            return torch.concatenate(self.comm.allgather(local_data))

        raise NotImplementedError(f"Input format '{input_format}' not implemented")
    
    
    def _balance_and_allgather(self, coo_reduced_region_topk, global_th):
        """
        Second main phase of ok_sparse_allreduce.  
        Performs the allgather of the coo_reduced_region_topk values among workers.

        Parameters:
            coo_reduced_region_topk (SparseTensorCOO): The sparse gradient tensor.
            global_th (float): the global threshold to perfrom top selection.

        Returns:
            out (tuple with two elements:):
                - coo_allgather_topk (SparseTensorCOO): A sparse gradient tensor with the global top-k selection.
                - reduced_region_global_topk_indexes (torch.Tensor): The indices of the top-k gradient values region reduced.
        """
        
        # 1. Global topk selection
        coo_reduced_region_global_topk = coo_reduced_region_topk.top_selection(global_th, inplace=False) 

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        coo_allgather_topk = self._allgather(coo_reduced_region_global_topk)
        return coo_allgather_topk, coo_reduced_region_global_topk.indexes
    
    
    def _intersect_indexes(self, local_topk_indexes, global_topk_indexes, method="numpy"):
        """
        Calculates the intersection of two sets of indices of 1D.

        Parameters:
            local_indexes (torch.Tensor): local indices sorted. 
            global_indexes (torch.Tensor): global indices sorted. 
        
        Returns:
            intersected_indexes (torch.Tensor): The intersection of indices.
        
        Example:
            - local_indexes  = torch.Tensor([0, 1, 2, 4])
            - global_indexes = torch.Tensor([1, 4, 6, 7, 9])
            - output: torch.Tensor([1, 4])  
        """
        
        self._show_message_only_once(f"In '_intersect_indexes', the method that it is being used is '{method}'")

        if method == "pytorch":
            combined = torch.cat((local_topk_indexes, global_topk_indexes))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            return intersection
        
        elif method == "numpy":
            local_i, global_i, intersect_i = 0, 0, 0
            max_intersection_size = min(len(local_topk_indexes), len(global_topk_indexes))
            intersect_topk_indexes = np.zeros(max_intersection_size, dtype=np.int64)
            while local_i < max_intersection_size and global_i < max_intersection_size:    
                if local_topk_indexes[local_i] == global_topk_indexes[global_i]:
                    intersect_topk_indexes[intersect_i] = local_topk_indexes[local_i]
                    intersect_i += 1
                    global_i += 1 
                    local_i += 1
                elif local_topk_indexes[local_i] < global_topk_indexes[global_i]:
                    local_i += 1
                else:
                    global_i += 1
            return intersect_topk_indexes[:intersect_i + 1]
            
        
    def _reduce_topk(self, coo_topk, boundaries, method="p2p_region_wise_reduce_static_destination"):
        """
        Reduce the topk elements in regions defined by boundaries.

        Parameters:
            coo_topk (SparseTensorCOO): a sparse tensor in COO format with the values and indexes of topk.
            boundaries (tensor.Torch): boundaries for partitioning the gradient space like [end_p0, end_p1, end_p2, ...].
            method (str, optional): The method to use for reduce topk

        Returns:
            coo_reduced_region (SparseTensorCOO): The reduced topk values in COO format.
        """
        
        self._show_message_only_once(f"In '_reduce_topk', the method that it is being used is '{method}'")
        
        if self.comm.size == 1:
            return coo_topk

        if method == "collective_allreduce_then_slice":
            warnings.warn("This reduce_topk method ('collective_allreduce_then_slice') should be used only in case of debugging for performance reasons.")
            all_reduced_coo = self.comm.allreduce(coo_topk, op=MPI.SUM)
            region_start = 0 if self.comm.rank == 0 else boundaries[self.comm.rank - 1]
            end = boundaries[self.comm.rank]
            return all_reduced_coo.slice(region_start, end)

        if method == "collective_region_wise_reduce_sync":
            region_start = 0
            reduced_regions_coo = [None] * self.comm.size
            for region in range(self.comm.size):
                region_end = boundaries[region]
                reduced_regions_coo[region] = self.comm.reduce(coo_topk.slice(region_start, region_end), op=MPI.SUM, root=region)
                region_start = region_end
            return reduced_regions_coo[self.comm.rank]

        if method == "p2p_region_wise_reduce_static_destination": # Ok-Topk-SP
            # Prepare a vector region for storing the partial sums
            coo_region_partial_sum = [None] * self.comm.size
            for region in range(self.comm.size):
                region_start = 0 if region == 0 else boundaries[region - 1]
                region_end = boundaries[region]
                coo_region_partial_sum[region] = coo_topk.slice(region_start, region_end)

            # Overlaps comm. steps with computation (sparse sum)
            # On comm_step i: P{rank} sends to P{rank + 1} region{rank - i % comm.size}. 
            destination = (self.comm.rank + 1) % self.comm.size
            receive_from = (self.comm.rank - 1) % self.comm.size
            for comm_step in range(1, self.comm.size):
                region_to_send = (self.comm.rank - comm_step) % self.comm.size
                region_to_recv = (self.comm.rank - comm_step - 1) % self.comm.size 
                coo_region_partial_sum[region_to_recv] += self.comm.sendrecv(coo_region_partial_sum[region_to_send], 
                                                                             dest=destination, source=receive_from)
            
            return coo_region_partial_sum[self.comm.rank]  
        
        
        if method == "all2all":
            # Prepare a vector region for storing the partial sums
            coo_region_partial_sum = [None] * self.comm.size
            for region in range(self.comm.size):
                region_start = 0 if region == 0 else boundaries[region - 1]
                region_end = boundaries[region]
                coo_region_partial_sum[region] = coo_topk.slice(region_start, region_end)
            
            my_region = coo_region_partial_sum[self.comm.rank]
            coo_region_partial_sum[self.comm.rank] = None
            coo_region = self.comm.alltoall(coo_region_partial_sum)
            coo_region[self.comm.rank] = my_region
            return sum(coo_region)
            
        
        if method == "p2p_region_wise_reduce_destination_rotation_and_bucketing": # Ok-Topk original
            # There are (nprocs - 1) messages to send (excluding self)
            total_sends = self.comm.size - 1
            requests = [None] * total_sends

            # Compute local slice of coo_topk (the "self" region)
            region_start = torch.tensor(0) if self.comm.rank == 0 else boundaries[self.comm.rank - 1]
            region_end = boundaries[self.comm.rank]
            coo_reduced_region = coo_topk.slice(region_start, region_end)

            # Process sends and receives in buckets.
            bucket_size = 2  
            region = (self.comm.rank + 1) % self.comm.size
            for comm_step in range(0, total_sends, bucket_size):
                # The current bucket may have fewer messages than bucket_size (i.e. the last bucket)
                current_bucket_size = min(bucket_size, total_sends - comm_step)
                # Non-blocking sends for the current bucket
                for i in range(current_bucket_size):
                    region_start = 0 if region == 0 else boundaries[region - 1]
                    region_end = boundaries[region]
                    requests[comm_step + i] = self.comm.isend(coo_topk.slice(region_start, region_end), dest=region)
                    region = (region + 1) % self.comm.size
                # After sending the bucket, perform the receives sequentially for the same bucket.
                for i in range(current_bucket_size):
                    coo_reduced_region += self.comm.recv()

            MPI.Request.Waitall(requests)
            return coo_reduced_region

        raise NotImplementedError(f"Method '{method}' not implemented")
    
    
    def zero_grad(self):
        for group in self.param_groups:
            for layer_params in group["params"]:
                if layer_params.grad is not None:
                    layer_params.grad.zero_()
                    

    def synchronize_grads(self):
        """
        Averages the model gradients across all processes using MPI.

        Args:
            model (torch.nn.Module): the model with the gradients already calculated (after backward).
            comm (MPI.Comm): MPI communicator (typically MPI.COMM_WORLD).
        
        TODO: 
            Investigar si es mejor las comunicaciones directamente con torch.distributed  en vez de MPI a pelo
        
        WARNING: 
            Es posible que esto se rompa al pasar a GPU, ya que no podrá asegurar que sean arrays contiguos
        """
        
        if self.comm.size == 1:
            return

        for group in self.param_groups:
            for layer_params in group["params"]:
                if layer_params.grad is not None:
                    self.comm.Allreduce(MPI.IN_PLACE, layer_params.grad.data, op=MPI.SUM)
                    layer_params.grad.data /= self.comm.size
        
    
    def _show_message_only_once(self, message):
        """
        Show information messages only once to assess the selected functions are being used.
        
        Parameters:
            message (str): The message to show.
        Returns:
            void (None):
        """
        if self.comm.rank == 0:
            if message not in self.info_messages:
                self.info_messages.add(message)
                print(message)