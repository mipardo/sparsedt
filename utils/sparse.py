import torch
import warnings


class SparseTensorCOO:
    """
    Represents a sparse tensor in COO format.
    This format stores the tensor using two arrays:
        - values: the nonzero values.
        - indexes: the indices corresponding to each value.
    The sparse tensor is assumed to be sorted.
    This class is not designed to store explict zeros so, len(self.values) should always be equal to nnz. 
    """

    def __init__(self, values, indexes, shape, has_canonical_format):
        """
        Primary initializer for SparseTensorCOO.
        
        Parameters:
            values (torch.tensor): Array with the nonzero values.
            indexes (torch.tensor): Array with the row indices.
            shape (tuple): Shape of the original tensor.
        """

        if len(values) != len(indexes):
            raise AssertionError("Values and indexes must have the same length")

        if indexes.dtype != torch.int64:
            raise AssertionError("Indexes type must be torch.int64, not", (indexes.dtype))
        
        if has_canonical_format:
            self.values = values
            self.indexes = indexes
            self.shape = shape
            self.nnz = len(self.values)
            self.has_canonical_format = True
            # assert(self._has_canonical_format())

        else:
            # TODO: order arrays in canonical format
            raise NotImplementedError("Not yet implemented constructor with unordered indexes")


    @classmethod
    def from_dense(cls, tensor):
        """
        Alternative constructor to create a SparseTensorCOO from a dense tensor.
        Only stores non-zero values!

        Parameters:
            tensor (torch.Tensor): A dense tensor.
        
        Returns:
            SparseTensorCOO: The sparse tensor in COO format
        """

        warnings.warn("From dense constructor should be used only in case of debugging for performance reasons.")

        indexes = torch.where(tensor != 0)
        values = tensor[indexes]
        return cls(values, indexes, tensor.shape, has_canonical_format=True)
        

    @classmethod
    def from_dense_top_selection(cls, tensor, threshold):
        """
        Alternative constructor to create a SparseTensorCOO from a dense tensor,
        considering only elements with absolute value greater than or equal to the threshold.
        
        Parameters:
            tensor (torch.tensor): A dense tensor.
            threshold (float): Threshold for including an element.
        
        Returns:
            SparseTensorCOO: The sparse tensor in COO format, containing only significant elements.
        """
        
        flattened_tensor = tensor.flatten()
        indexes = torch.where(flattened_tensor.abs() >= threshold)[0]
        topk = flattened_tensor[indexes]
        return cls(topk, indexes, tensor.shape, has_canonical_format=True)
    

    def top_selection(self, threshold, inplace=True):
        """
        Performs top threshold selection on sparse array

        Parameters:
            threshold (float): Threshold for including an element.
            inplace (bool, optional): 

        Returns:
            topk (SparseTensorCOO): if inplace == False, or void (None): if inplace == True 
        """

        mask = self.values >= threshold
        topk_values = self.values[mask]
        topk_indexes = self.indexes[mask]

        if inplace:
            self.values = topk_values
            self.indexes = topk_indexes
            self.nnz = topk_values.numel()
            # self.shape remains equal
            # self.has_canonical_format remains equal
        else:
            return SparseTensorCOO(topk_values, topk_indexes, self.shape, self.has_canonical_format)
        

    def slice(self, start, end, reset_indexes=False):
        """
        Perform a slice of the sparse tensor.
        
        Parameters:
            start (int): The starting index (inclusive) of the slice.
            end (int): The ending index (exclusive) of the slice.
            reset_indexes (bool, optional): If True, resets the indices of the 
                                            sliced tensor so that "start" maps to zero.
                                            Defaults to False.
        Returns:
            coo_sliced: (SparseTensorCOO): A sliced sparse tensor of self
        """

        sliced_indexes = self.indexes[start:end] - start if reset_indexes else self.indexes[start:end]
        return SparseTensorCOO(self.values[start:end], sliced_indexes, self.shape, self.has_canonical_format)


    def to_dense(self):
        """
        Convert to dense np.array
        
        Returns:
            dense_tensor: (np.array): a dense tensor
        """

        warnings.warn("This function ('to_sparse') should be used only in case of debugging for performance reasons.")

        dense_tensor = torch.zeros(self.shape, dtype=torch.float32)
        dense_tensor[torch.unravel_index(self.indexes, self.shape)] = self.values
        return dense_tensor
    

    def __add__(self, other):
        """
        Adds two SparseTensorCOO that are in canonical format.
        
        Parameters:
            other (SparseTensorCOO): Another SparseTensorCOO instance.
        
        Returns:
            SparseTensorCOO: A new instance representing the sum of both tensor.
        """

        if type(other) != SparseTensorCOO:
            raise AssertionError("Operand must be a SparseTensorCOO instance.")
        if self.shape != other.shape:
            raise AssertionError("Tensors must have the same shape.")
        if not self.has_canonical_format or not other.has_canonical_format:
            raise AssertionError("Both matrices must be in canonical format.")


        self_i, other_i, summ_i = 0, 0, 0
        summ_values = torch.zeros(self.nnz + other.nnz)
        summ_indexes = torch.zeros(self.nnz + other.nnz, dtype=torch.int64)
        while self_i < self.nnz and other_i < other.nnz:    
            if self.indexes[self_i] == other.indexes[other_i]:
                summ_values[summ_i] = self.values[self_i] + other.values[other_i]
                summ_indexes[summ_i] = self.indexes[self_i]
                other_i += 1 
                self_i += 1
                summ_i += 1
            elif self.indexes[self_i] < other.indexes[other_i]:
                summ_values[summ_i] = self.values[self_i]
                summ_indexes[summ_i] = self.indexes[self_i]
                self_i += 1
                summ_i += 1
            else:
                summ_values[summ_i] = other.values[other_i]
                summ_indexes[summ_i] = other.indexes[other_i]
                other_i += 1
                summ_i += 1

        return SparseTensorCOO(summ_values[:summ_i], summ_indexes[:summ_i], self.shape, has_canonical_format=True)
    

    def __radd__(self, other):
        """
        Implements right-hand addition to support the built-in sum() function.
        
        This method allows an instance of this class to be used with sum() by handling the
        case where the left operand is 0. If 'other' is 0, it returns the instance itself;
        otherwise, it delegates the operation to the __add__ method.
        
        Parameters:
            other (int or instance of the same class): The left-hand operand, typically 0 when used with sum().
            
        Returns:
            An instance of the class representing the sum of self and other.
        """
        if other == 0:
            return self
        return self.__add__(other)


    def __repr__(self):
        return f"SparseTensorCOO(values={self.values}, indexes={self.indexes}, shape={self.shape}, nnz={self.nnz})"


    def _has_canonical_format(self):
        """
        Check if SparseTensorCOO follows canonical format: 
            - Indexes are sorted
            - There are no duplicate entries
        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format. 

        Returns:
            has_canonical_format (boolean): True if indexes are in canonical format, False if not. 
        """
        
        warnings.warn("This function ('has_canonical_format') should be used only in case of debugging for performance reasons.")
        
        if self.nnz == 0:
            return True

        for i in range(self.nnz - 1):
            if self.indexes[i] >= self.indexes[i + 1]:
                return False
        return True
    

        