# ml-monocular-distance-estimation
A machine learning-based non-linear regression model to accurately predict the real-world distance of an object using images captured from a single (monocular) camera. The project leverages camera calibration, pixel-size analysis, and temporal tracking techniques to provide cost-effective, accurate distance estimation.


# Relationship
Trying to train a model to learn and accurately apply the mathmatical relationship between an objects pixel size and its real diameter. This can be represented as:

distance = (Focal Length(mm) * Real Diameter(mm)) / (Pixel Diameter * Pixel Size(mm/pixel))
