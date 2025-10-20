from mpi4py import MPI
from torch.optim import Optimizer


class SSGD(Optimizer):
    def __init__(self, model_params, lr, momentum, weight_decay):
        self.comm = MPI.COMM_WORLD
        super().__init__(model_params, defaults={"lr": lr, "momentum":momentum, "weight_decay": weight_decay})
        
        
    def step(self):
        self.synchronize_grads()
        for group in self.param_groups:
            for layer_params in group['params']:        
                if layer_params.grad is not None:
                    grads = layer_params.grad
                    # Apply weight decay
                    if group['weight_decay'] != 0:
                        grads = grads.add(layer_params.data, alpha=group['weight_decay'])
                    # Apply momentum
                    if group['momentum'] != 0:
                        state = self.state[layer_params]
                        if 'momentum_buffer' not in state:
                            buf = grads.clone()
                            state['momentum_buffer'] = buf
                        else:
                            buf = state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(grads)
                        grads = buf
                    # Update parameters
                    layer_params.data.add_(grads, alpha=-group['lr'])


    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
        
    
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
            for p in group['params']:
                if p.grad is not None:
                    self.comm.Allreduce(MPI.IN_PLACE, p.grad.data, op=MPI.SUM)
                    p.grad.data /= self.comm.size