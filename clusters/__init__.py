"""Clustering lab: k-means (k-means++ seeding), DBSCAN, and a Gaussian mixture fitted by EM.

Everything is seeded and uses one shared PRNG (mulberry32), so the browser page (web/js/clusters-core.js)
draws the same presets, seeds and EM starts as this package. Tests check the two against each other.
"""
