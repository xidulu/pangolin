Pangolin is a probabilistic inference research project. It has the following goals:

* **Feel like numpy.** Provide an interface for interacting with probability 
  distributions that feels natural to anyone who's played around with 
  numpy. As much as possible, using Pangolin should feel like a natural extension of 
  that ecosystem rather than a new language.
* **Look like math.** Where possible, calculations should resemble mathematical 
  notation. Exceptions are allowed when algorithmic limitations make this impossible or 
  where accepted mathematical notation is bad.
* **Gradual enhancement.** There should be no barrier between using it as a 
  "calculator" for simple one-liner probabilistic calculations and building full 
  custom Bayesian models.
* **Multiple Backends.** We have used lots of different PPLs. Some of our favorites are:
  * [BUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/openbugs)
  * [JAGS](https://mcmc-jags.sourceforge.io/)
  * [Stan](https://mc-stan.org/)
  * [NumPyro](https://num.pyro.ai/)
  * [PyMC](https://www.pymc.io/)
  * [Tensorflow Probability](https://www.tensorflow.org/probability)  
   We want users to be able to write a model *once* and then seamlessly use any of 
    these to actually do inference.  
* **Support program transformations.** Often, users of different PPLs need to 
  manually "transform" their model to get good results. (E.g. manually integrating out 
  discrete latent variables, using non-centered transformations, etc.) We want to 
  provide an "intermediate representation" to make such transformations as easy to 
  define as possible.
* **Support inference algorithm developers.** Existing probabilistic programming 
  languages are often quite "opinionated" about how inference should proceed. This 
  can make it difficult to apply certain kinds of inference algorithms.
  We want to provide a representation that makes it as easy as possible for people 
  to create "new" backends with new inference algorithms.
* **No passing strings as variable names.** We love [NumPyro](https://num.pyro.ai/) 
  and [PyMC](https://www.pymc.io/). But we don't love writing `mean = sample('mean',
  Normal(0, 1))` or `mean = Normal('mean', 0, 1)`. And we *really* don't love 
  programmatically generating variable name strings inside of a for loop. We 
  appreciate that Tensorflow Probability made this mostly optional, but we feel this 
  idea was a mistake and we aren't going to be so compromising.

It remains to be seen to what degree all these goals can be accomplished at the same 
time. (That's what makes this a research project!)

At the moment, we **do not advise** trying to use this code. However, an earlier 
version of Pangolin is available and based on much the same ideas, except only 
supporting JAGS as a backend. It can be found with documentation, in the 
[`pangolin-jags`](pangolin-jags) directory.