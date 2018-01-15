#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:38:39 2017

@author: chunyilyu
"""
import numpy as np
from numbers import Number
import math
#from pyquaternion.quaternion import Quaternion as tQuaternion
class Quaternion:
    def __init__(self, scalar, vector, print_as_ijk=False):

        self.s = scalar
        self.v = np.array(vector)
        self.print_as_ijk = print_as_ijk

    def __str__(self):
        if self.print_as_ijk:
            words = []
            labels = ["1", "i", "j", "k"]
            words.append(str(self.s))
            for i in range(0, 3):
                if self.v[i] >= 0:
                    words.append("+")
                words.append(str(self.v[i]))
                words.append(labels[i])
            return "".join(words)
        #return "{:.3f} {:+.3f}i {:+.3f}j {:+.3f}k".format(self.s, self.v[0], self.v[1], self.v[2])
        else:
            return str([self.s, self.v.tolist()])

    def __repr__(self):
        return "Quaternion({},{})".format(self.s, self.v.tolist())

    def __neg__(self):
        return Quaternion(-self.s, -self.v, print_as_ijk=self.print_as_ijk)

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.s + other.s, self.v + other.v, print_as_ijk=self.print_as_ijk)
        else:
            return Quaternion(self.s + other, self.v, print_as_ijk=self.print_as_ijk)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.s - other.s, self.v - other.v, print_as_ijk=self.print_as_ijk)
        else:
            return Quaternion(self.s - other, self.v, print_as_ijk=self.print_as_ijk)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        assert isinstance(other, Number) or isinstance(other, Quaternion) \
                                            and "multiplication only defined for scalars and quaternions"
        if isinstance(other, Quaternion):
            return self.multiply(other)
        else:
            return Quaternion(other*self.s, other*self.v, print_as_ijk=self.print_as_ijk)

    def __rmul__(self, other):
        assert isinstance(other, Number) or isinstance(other, Quaternion) \
                                            and "multiplication only defined for scalars and quaternions"

        if isinstance(other, Quaternion):
            return other.multiply(self)
        else:
            return Quaternion(other*self.s, other*self.v, print_as_ijk=self.print_as_ijk)

    def __div__(self, other):
        assert isinstance(other, Number) and "division only defined for quaternions"
        return Quaternion(self.s/other, self.v/other)

    def to_numpy(self):
        return np.hstack((self.s, self.v))

    def to_list(self):
        return [self.s] + self.v.tolist()

    def multiply(self, quaternion):
        return Quaternion(self.s*quaternion.s - self.v.T.dot(quaternion.v),
                          self.s*quaternion.v + quaternion.s*self.v + np.cross(self.v, quaternion.v),
                          print_as_ijk=self.print_as_ijk)

    def conjugate(self):
        return Quaternion(self.s, -self.v, print_as_ijk=self.print_as_ijk)

    def norm(self):
        return np.sqrt(self.s**2 + self.v.T.dot(self.v)) + 1e-20

    def inv(self):
        return Quaternion(self.s/self.norm()**2, -self.v/self.norm()**2, print_as_ijk=self.print_as_ijk)

    def unit(self):
        return self/self.norm()

    def rotate_rotation_vector(self, vector):
        return self.multiply(Quaternion(0, vector)).multiply(self.conjugate()).v

    def rotate_vector(self, vector):
        return self.multiply(Quaternion(0, vector)).multiply(self.inv()).v

    def rotate_quaternion(self, quaternion):
        return self.multiply(quaternion).multiply(self.inv())

    def exp(self):
        if np.all(self.v == 0):
            return np.exp(self.s)*Quaternion(np.cos(np.linalg.norm(self.v)), [0, 0, 0])
        else:
          
            return np.exp(self.s)*Quaternion(np.cos(np.linalg.norm(self.v)),
                                             self.v/np.linalg.norm(self.v)*np.sin(np.linalg.norm(self.v)),
                                             print_as_ijk=self.print_as_ijk)

    def log(self):
        if np.all(self.v == 0):
            return Quaternion(np.log(self.norm()), [0, 0, 0])
        else:
            return Quaternion(np.log(self.norm()),
                              self.v/np.linalg.norm(self.v)*np.arccos(self.s/self.norm()),
                              print_as_ijk=self.print_as_ijk)
