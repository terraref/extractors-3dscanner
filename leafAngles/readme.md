## Code for calculating leaf angle distributions

Leaf angle distribution is a key parameter to characterize canopy structure and plays a crucial role in controlling energy. The laser scanner in TERRA-REF gantry system is creating high throughput high resolution 3d plant models. We use 3d plant data to compute leaf normal orientation and then leads to leaf angle distribution. At last we generate two-parameter beta distribution1 and chi-parameter2 as our ‘single value traits’.

The 3d point cloud data derived from 2d reflectance image and 2d depth data(a). Each pixel in 2d data corresponds to one point in 3d data. In order to obtain a set of points quickly, a region on interest(ROI) window move across the reflectance image. With a mapping matrix between reflectance image and 3d point cloud, target points can be extracted very quickly(d), the whole 3d data is divided into a huge number of small components. Figure (d) shows a set of points from the selected ROI window. Figure (e) is an abstract of figure (d), the blue plane  represents for leaf surface. When the ROI point cloud data extracted, Singular Value Decomposition(SVD) was applied to compute the leaf normal vector3. 

A color coded reflectance image is made to visualize the leaf normal orientation(figure b), different color represent for different orientation, then we should see a continuous 'rainbow' color on leaves. 

The next step is to create one single value to describe the leaf angle distribution. Two-parameter beta distribution and chi-parameter distribution are two widely used leaf angle parameters. Figure (c) is a plot of leaf angle distribution compared with initial curve function created by 'chi-distribution' and best fitted curve. x-axis is inclination to vertical from 0 degree to 90 degree, re-scaled to (0, 1), y-axis is a normalized number of how many points in that degree, normalized by 'area in x-axis and function lines equals to 1'. Blue dots are leaf angle distribution(LAD) from laser data using previous methods. Dotted line shows the initial LAD by source chi-parameter, and red line shows a updated LAD by the best fitted chi-parameter. It shows that the best-fitted chi-parameter is able to represent the whole LAD.

### References

Simple Beta Distribution Representation of Leaf Orientation in Vegetation Canopies, Narendra S. Goel, Donald E. Strebel, 1984

Derivation of an angle density function for canopies with ellipsoidal leaf angle distributions, G.S Campbell, 1990, Agricultural and Forest Meteorology  
https://en.wikipedia.org/wiki/Singular_value_decomposition

### Issues 

https://github.com/terraref/computing-pipeline/issues/338
