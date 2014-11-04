Spatial Interaction Modeling Code
==================================

This code provides a means for carry out the calibration of a 'family' of spatial interaction model (Fotheringham & O'Kelly, 1989) which are derived using an entropy maximizing (EM) framework or the equivalent information minimizing (IM) framework. As such, it is able to derive parameters for the following models:

- unconstrained gravity model
- production-constrained model (origin-constrained)
- attraction-constrained model (destination-constrained)
- doubly-constrained model


The original code was developed with the intention of building a more general framework for spatial interaction modeling. It supports the calibration of the above four models with either a power function or an exponential function of distance decay. It originally was designed so that it was possible to calibrate a model using maximum likelihood (ML) optimization framework or using a GLM regression framework, using the exact same inputs. The regression support has been removed for simplicity and what remains is the ML framework which leverages scipy.optimize.fsolve(). Overall, the routine is currently dependent upon pandas as well, though I think it could be re-written without it.

Fotheringham, A. S. and O'Kelly, M. E. (1989). Spatial Interaction Models: Formulations and Applications. London: Kluwer Academic Publishers.
