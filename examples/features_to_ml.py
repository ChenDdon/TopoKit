"""Numerical feature matrices; this example makes no prediction-accuracy claim."""
import numpy as np
from topokit import PointCloud
from topokit.builders import from_points
from topokit.core import persistence
from topokit.postprocessing import PersistenceVectorizer

points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
clouds = [PointCloud(points), PointCloud(points * 1.2)]
barcodes = [persistence(from_points(cloud), max_dimension=2) for cloud in clouds]
# Fixed schema: choose it from scientific requirements or training data only.
encoder = PersistenceVectorizer(birth_edges=np.linspace(0, 2, 9), death_edges=np.linspace(0, 2, 9))
X_train = encoder.fit_transform(barcodes[:1])
X_test = encoder.transform(barcodes[1:])
print("Training/test feature matrices:", X_train.shape, X_test.shape)
print("Ready for a downstream estimator once real targets and valid data splits exist.")
# Optional, once real training labels exist (no training is run by this example):
# from topokit.workflows.ml import fit_features
# model = fit_features(X_train, y_train)  # default: gradient-boosted regression trees
# y_pred = model.predict(X_test)
