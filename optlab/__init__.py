"""Optimizer race: gradient descent, momentum, Nesterov, RMSProp, Adam and AdaGrad on 2D loss landscapes.

This package is the Python reference for the browser lab (web/js/optlab-core.js). Surfaces give the loss
and its analytic gradient, the optimizers are written from scratch, and the race runs them all from one
start point with one shared gradient-noise stream. The tests check the gradients by finite differences and
compare the JavaScript port with this code to 1e-9.
"""
