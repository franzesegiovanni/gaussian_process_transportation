from policy_transportation import AffineTransform
import numpy as np
print("Test 2 d case")
source_distribution = np.random.rand(3, 2)
target_distribution = np.random.rand(3, 2)

affine_transform = AffineTransform()
affine_transform.fit(source_distribution, target_distribution)

print("Test 3 d case")
source_distribution = np.random.rand(4, 3)
target_distribution = np.random.rand(4, 3)

affine_transform = AffineTransform()

affine_transform.fit(source_distribution, target_distribution)
