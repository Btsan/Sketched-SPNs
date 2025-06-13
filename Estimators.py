import pandas as pd
import torch

class CountEstimator(object):
    def __init__(self, sketch_lo: torch.Tensor, sketch_hi: torch.Tensor = None, exact_hi: pd.Series = None):
        self.sketch_lo = sketch_lo
        self.shape = sketch_lo.shape

        # hi-freq estimation componenent for bifocal estimator
        if sketch_hi is not None and exact_hi is not None:
            self.is_bifocal = True
            self.exact_hi = exact_hi.copy()
            self.sketch_hi = sketch_hi
        else:
            self.is_bifocal = False
            self.exact_hi = None
            self.sketch_hi = None

    def cuda(self):
        new_sketch_lo = self.sketch_lo.cuda()
        if self.is_bifocal:
            new_sketch_hi = self.sketch_hi.cuda()
            return type(self)(new_sketch_lo, new_sketch_hi, self.exact_hi)
        return type(self)(new_sketch_lo)

    def memory_usage(self):
        nbytes = self.sketch_lo.numel() * self.sketch_lo.element_size()
        if self.is_bifocal:
            nbytes += self.exact_hi.memory_usage()
            nbytes += self.sketch_hi.numel() * self.sketch_hi.element_size()
        return nbytes
    
    def total(self):
        total = self.sketch_lo.abs().sum().item()
        if self.is_bifocal:
            total += self.exact_hi.abs().sum().item()
            total += self.sketch_hi.abs().sum().item()
        return total

    def __repr__(self):
        if self.is_bifocal:
            return f"sketch (infrequent) {self.sketch_lo.shape} {self.sketch_lo}\n" \
                +   f"sketch (frequent) {self.sketch_hi.shape} {self.sketch_hi}\n" \
                +   self.exact_hi.to_string(float_format='{:,.2f}'.format) \
                +   f"\nIndex={self.exact_hi.index.names}"
        return f"sketch (infrequent) {self.sketch_lo.shape} {self.sketch_lo}"

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_sketch_lo = self.sketch_lo + other.sketch_lo
            if not self.is_bifocal:
                new_exact_hi = other.exact_hi
                new_sketch_hi = other.sketch_hi
            elif not other.is_bifocal:
                new_exact_hi = self.exact_hi
                new_sketch_hi = self.sketch_hi
            else:
                new_exact_hi = self.exact_hi.add(other.exact_hi, fill_value=0)
                new_sketch_hi = self.sketch_hi + other.sketch_hi
            return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
        if isinstance(other, (int, float)):
            assert other == 0
            new_sketch_lo = self.sketch_lo + other
            if not self.is_bifocal:
                return type(self)(new_sketch_lo)
            
            new_exact_hi = self.exact_hi + other
            new_sketch_hi = self.sketch_hi + other
            return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.sketch_lo += other.sketch_lo
            if not self.is_bifocal:
                self.exact_hi = other.exact_hi
                self.sketch_hi = other.sketch_hi
            elif other.is_bifocal:
                self.exact_hi = self.exact_hi.add(other.exact_hi, fill_value=0)
                self.sketch_hi += other.sketch_hi
            return self
        elif isinstance(other, (int, float)):
            assert other == 0
            return self
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_sketch_lo = self.sketch_lo * other.sketch_lo
            if not self.is_bifocal:
                new_exact_hi = other.exact_hi
                new_sketch_hi = other.sketch_hi
            elif not other.is_bifocal:
                new_exact_hi = self.exact_hi
                new_sketch_hi = self.sketch_hi
            else:
                new_exact_hi = self.exact_hi.mul(other.exact_hi, fill_value=0)
                new_sketch_hi = self.sketch_hi * other.sketch_hi
            return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
        elif isinstance(other, (int, float)):
            new_sketch_lo = self.sketch_lo * other
            # new_sketch_lo[(-1 < new_sketch_lo) & (new_sketch_lo < 0)] = -1
            # new_sketch_lo[(0 < new_sketch_lo) & (new_sketch_lo < 1)] = 1
            if self.is_bifocal:
                new_exact_hi = self.exact_hi * other
                # new_exact_hi[(0 < new_exact_hi) & (new_exact_hi < 1)] = 1
                new_sketch_hi = self.sketch_hi * other
                # new_sketch_hi[new_sketch_hi < 1] = 1
                # new_sketch_hi[(-1 < new_sketch_hi) & (new_sketch_hi < 0)] = -1
                # new_sketch_hi[(0 < new_sketch_hi) & (new_sketch_hi < 1)] = 1
                return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
            return type(self)(new_sketch_lo)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        if isinstance(other, self.__class__):
            self.sketch_lo *= other.sketch_lo
            if not self.is_bifocal:
                self.exact_hi = other.exact_hi
                self.sketch_hi = other.sketch_hi
            elif other.is_bifocal:
                self.exact_hi = self.exact_hi.mul(other.exact_hi, fill_value=0)
                self.sketch_hi *= other.sketch_hi
            return self
        elif isinstance(other, (int, float)):
            self.sketch_lo *= other
            # self.sketch_lo[(-1 < self.sketch_lo) & (self.sketch_lo < 0)] = -1
            # self.sketch_lo[(0 < self.sketch_lo) & (self.sketch_lo < 1)] = 1
            if self.is_bifocal:
                self.exact_hi *= other
                # self.exact_hi[self.exact_hi < 1] = 1
                self.sketch_hi *= other
                # self.sketch_hi[(-1 < self.sketch_hi) & (self.sketch_hi< 0)] = -1
                # self.sketch_hi[(0 < self.sketch_hi) & (self.sketch_hi < 1)] = 1
            return self
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            new_sketch_lo = self.sketch_lo / other.sketch_lo
            if not self.is_bifocal:
                new_exact_hi = other.exact_hi
                new_sketch_hi = other.sketch_hi
            elif not other.is_bifocal:
                new_exact_hi = self.exact_hi
                new_sketch_hi = self.sketch_hi
            else:
                new_exact_hi = self.exact_hi.div(other.exact_hi, fill_value=1)
                new_sketch_hi = self.sketch_hi / other.sketch_hi
            return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
        elif isinstance(other, (int, float)):
            new_sketch_lo = self.sketch_lo / other
            if self.is_bifocal:
                new_exact_hi = self.exact_hi / other
                new_sketch_hi = self.sketch_hi / other
                return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
            return type(self)(new_sketch_lo)
        return NotImplemented
    
    def __itruediv__(self, other):
        if isinstance(other, self.__class__):
            self.sketch_lo /= other.sketch_lo
            if not self.is_bifocal:
                self.exact_hi = other.exact_hi
                self.sketch_hi = other.sketch_hi
            elif other.is_bifocal:
                self.exact_hi = self.exact_hi.div(other.exact_hi, fill_value=1)
                self.sketch_hi /= other.sketch_hi
            return self
        elif isinstance(other, (int, float)):
            self.sketch_lo /= other
            if self.is_bifocal:
                self.exact_hi /= other
                self.sketch_hi /= other
            return self
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.total() < other
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.total() <= other
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.total() > other
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.total() >= other
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.total() == other
        return NotImplemented

class DegreeEstimator(CountEstimator):

    ### proposed element-wise maximum results in underestimation
    # def __add__(self, other):
    #     if isinstance(other, self.__class__):
    #         new_sketch_lo = torch.maximum(self.sketch_lo, other.sketch_lo)
    #         if not self.is_bifocal:
    #             new_exact_hi = other.exact_hi
    #             new_sketch_hi = other.sketch_hi
    #         elif not other.is_bifocal:
    #             new_exact_hi = self.exact_hi
    #             new_sketch_hi = self.sketch_hi
    #         else:
    #             new_exact_hi = self.exact_hi.add(other.exact_hi, fill_value=0)
    #             new_sketch_hi = torch.maximum(self.sketch_hi, other.sketch_hi)
    #         return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
    #     if isinstance(other, (int, float)):
    #         assert other == 0
    #         new_sketch_lo = self.sketch_lo + other
    #         if not self.is_bifocal:
    #             return type(self)(new_sketch_lo)
            
    #         new_exact_hi = self.exact_hi + other
    #         new_sketch_hi = self.sketch_hi + other
    #         return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
    #     return NotImplemented
    
    # def __iadd__(self, other):
    #     if isinstance(other, self.__class__):
    #         self.sketch_lo = torch.maximum(self.sketch_lo, other.sketch_lo)
    #         if not self.is_bifocal:
    #             self.exact_hi = other.exact_hi
    #             self.sketch_hi = other.sketch_hi
    #         elif other.is_bifocal:
    #             self.exact_hi = self.exact_hi.add(other.exact_hi, fill_value=0)
    #             self.sketch_hi = torch.maximum(self.sketch_hi, other.sketch_hi)
    #         return self
    #     elif isinstance(other, (int, float)):
    #         assert other == 0
    #         return self
    #     return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_sketch_lo = self.sketch_lo * other.sketch_lo
            if not self.is_bifocal:
                new_exact_hi = other.exact_hi
                new_sketch_hi = other.sketch_hi
            elif not other.is_bifocal:
                new_exact_hi = self.exact_hi
                new_sketch_hi = self.sketch_hi
            else:
                new_exact_hi = self.exact_hi.mul(other.exact_hi, fill_value=0)
                new_sketch_hi = self.sketch_hi * other.sketch_hi
            return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
        elif isinstance(other, (int, float)):
            if other != 0:
                # do not scale, just copy self
                new_sketch_lo = self.sketch_lo
            else:
                new_sketch_lo = self.sketch_lo * other
            if self.is_bifocal:
                if other != 0:
                    # do not scale, just copy self
                    new_exact_hi = self.exact_hi
                    new_sketch_hi = self.sketch_hi
                else:
                    new_exact_hi = self.exact_hi * other
                    new_sketch_hi = self.sketch_hi * other
                return type(self)(new_sketch_lo, new_sketch_hi, new_exact_hi)
            return type(self)(new_sketch_lo)
        return NotImplemented
    
    def __imul__(self, other):
        if isinstance(other, self.__class__):
            self.sketch_lo *= other.sketch_lo
            if not self.is_bifocal:
                self.exact_hi = other.exact_hi
                self.sketch_hi = other.sketch_hi
            elif other.is_bifocal:
                self.exact_hi = self.exact_hi.mul(other.exact_hi, fill_value=0)
                self.sketch_hi *= other.sketch_hi
            return self
        elif isinstance(other, (int, float)):
            if other == 0:
                self.sketch_lo *= 0
                if self.is_bifocal:
                    self.exact_hi *= 0
                    self.sketch_hi *= 0
            return self
        return NotImplemented