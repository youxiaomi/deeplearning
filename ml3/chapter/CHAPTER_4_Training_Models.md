## **CHAPTER 4 Training Models**

So far we have treated machine learning models and their training algorithms mostly like black boxes. If you went through some of the exercises in the previous chapters, you may have been surprised by how much you can get done without knowing anything about what's under the hood: you optimized a regression system, you improved a digit image classifier, and you even built a spam classifier from scratch, all without knowing how they actually work. Indeed, in many situations you don't really need to know the implementation details.

However, having a good understanding of how things work can help you quickly home in on the appropriate model, the right training algorithm to use, and a good set of hyperparameters for your task. Understanding what's under the hood will also help you debug issues and perform error analysis more efficiently. Lastly, most of the topics discussed in this chapter will be essential in understanding, building, and training neural networks (discussed in Part II of this book).

In this chapter we will start by looking at the linear regression model, one of the simplest models there is. We will discuss two very different ways to train it:

- Using a "closed-form" equation<sup>1</sup> that directly computes the model parameters that best fit the model to the training set (i.e., the model parameters that minimize the cost function over the training set).
- Using an iterative optimization approach called gradient descent (GD) that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as the first method. We will look at a few variants of gradient descent that we will use again and

<sup>1</sup> A closed-form equation is only composed of a finite number of constants, variables, and standard operations: for example,  $a = \sin(b - c)$ . No infinite sums, no limits, no integrals, etc.

{159}------------------------------------------------

again when we study neural networks in Part II: batch GD, mini-batch GD, and stochastic GD

Next we will look at polynomial regression, a more complex model that can fit nonlinear datasets. Since this model has more parameters than linear regression, it is more prone to overfitting the training data. We will explore how to detect whether or not this is the case using learning curves, and then we will look at several regularization techniques that can reduce the risk of overfitting the training set.

Finally, we will examine two more models that are commonly used for classification tasks: logistic regression and softmax regression.

![](img/_page_159_Picture_3.jpeg)

There will be quite a few math equations in this chapter, using basic notions of linear algebra and calculus. To understand these equations, you will need to know what vectors and matrices are; how to transpose them, multiply them, and inverse them; and what partial derivatives are. If you are unfamiliar with these concepts, please go through the linear algebra and calculus introductory tutorials available as Jupyter notebooks in the online supplemental material. For those who are truly allergic to mathematics, you should still go through this chapter and simply skip the equations; hopefully, the text will be sufficient to help you understand most of the concepts.

### **Linear Regression**

In Chapter 1 we looked at a simple regression model of life satisfaction:

```
life_satisfaction = \theta_0 + \theta_1 \times GDP\_per\_capita
```

This model is just a linear function of the input feature GDP\_per\_capita.  $\theta_0$  and  $\theta_1$  are the model's parameters.

More generally, a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept *term*), as shown in Equation 4-1.

```
Equation 4-1. Linear regression model prediction
```

 $\widehat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$ 

In this equation:

- $\hat{y}$  is the predicted value.
- $\bullet$  *n* is the number of features.

{160}------------------------------------------------

- $x_i$  is the *i*<sup>th</sup> feature value.
- $\theta_i$  is the j<sup>th</sup> model parameter, including the bias term  $\theta_0$  and the feature weights  $\theta_1$ ,  $\theta_2, \dots, \theta_n$

This can be written much more concisely using a vectorized form, as shown in Equation 4-2.

Equation 4-2. Linear regression model prediction (vectorized form)

 $\hat{y} = h_{\mathbf{\Theta}}(\mathbf{x}) = \mathbf{\Theta} \cdot \mathbf{x}$ 

In this equation:

- $h_{\theta}$  is the hypothesis function, using the model parameters  $\theta$ .
- $\theta$  is the model's *parameter vector*, containing the bias term  $\theta_0$  and the feature weights  $\theta_1$  to  $\theta_n$ .
- **x** is the instance's *feature vector*, containing  $x_0$  to  $x_n$ , with  $x_0$  always equal to 1.
- $\theta$  · **x** is the dot product of the vectors  $\theta$  and **x**, which is equal to  $\theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2$  $+ ... + \theta_{n} x_{n}$

![](img/_page_160_Picture_10.jpeg)

In machine learning, vectors are often represented as column vectors, which are 2D arrays with a single column. If  $\theta$  and x are column vectors, then the prediction is  $\hat{y} = \theta^T x$ , where  $\theta^T$  is the *transpose* of  $\theta$  (a row vector instead of a column vector) and  $\theta^{\mathsf{T}}\mathbf{x}$ is the matrix multiplication of  $\theta^{\dagger}$  and **x**. It is of course the same prediction, except that it is now represented as a single-cell matrix rather than a scalar value. In this book I will use this notation to avoid switching between dot products and matrix multiplications.

OK, that's the linear regression model—but how do we train it? Well, recall that training a model means setting its parameters so that the model best fits the training set. For this purpose, we first need a measure of how well (or poorly) the model fits the training data. In Chapter 2 we saw that the most common performance measure of a regression model is the root mean square error (Equation 2-1). Therefore, to train a linear regression model, we need to find the value of  $\theta$  that minimizes the RMSE. In practice, it is simpler to minimize the mean squared error (MSE) than the RMSE, and it leads to the same result (because the value that minimizes a positive function also minimizes its square root).

{161}------------------------------------------------

![](img/_page_161_Picture_0.jpeg)

Learning algorithms will often optimize a different loss function during training than the performance measure used to evaluate the final model. This is generally because the function is easier to optimize and/or because it has extra terms needed during training only (e.g., for regularization). A good performance metric is as close as possible to the final business objective. A good training loss is easy to optimize and strongly correlated with the metric. For example, classifiers are often trained using a cost function such as the log loss (as you will see later in this chapter) but evaluated using precision/recall. The log loss is easy to minimize, and doing so will usually improve precision/recall.

The MSE of a linear regression hypothesis  $h_{\theta}$  on a training set **X** is calculated using Equation 4-3.

Equation 4-3. MSE cost function for a linear regression model

$$
MSE(\mathbf{X}, h_{\boldsymbol{\theta}}) = \frac{1}{m} \sum_{i=1}^{m} (\boldsymbol{\theta}^{\mathsf{T}} \mathbf{x}^{(i)} - y^{(i)})^2
$$

Most of these notations were presented in Chapter 2 (see "Notations" on page 44). The only difference is that we write  $h_{\theta}$  instead of just h to make it clear that the model is parametrized by the vector  $\theta$ . To simplify notations, we will just write  $MSE(\theta)$ instead of MSE( $X$ ,  $h_a$ ).

### **The Normal Equation**

To find the value of  $\theta$  that minimizes the MSE, there exists a *closed-form solution*—in other words, a mathematical equation that gives the result directly. This is called the Normal equation (Equation 4-4).

Equation 4-4. Normal equation

$$
\widehat{\boldsymbol{\theta}} = \left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \quad \mathbf{X}^{\top} \quad \mathbf{y}
$$

In this equation:

- $\theta$  is the value of  $\theta$  that minimizes the cost function.
- y is the vector of target values containing  $y^{(1)}$  to  $y^{(m)}$ .

{162}------------------------------------------------

Let's generate some linear-looking data to test this equation on (Figure  $4-1$ ):

![](img/_page_162_Figure_1.jpeg)

Figure 4-1. A randomly generated linear dataset

Now let's compute  $\hat{\theta}$  using the Normal equation. We will use the inv() function from NumPy's linear algebra module (np.linalg) to compute the inverse of a matrix, and the dot() method for matrix multiplication:

```
from sklearn.preprocessing import add_dummy_feature
X b = add dummy feature(X) # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

![](img/_page_162_Picture_5.jpeg)

The @ operator performs matrix multiplication. If A and B are NumPy arrays, then A @ B is equivalent to np.matmul(A, B). Many other libraries, like TensorFlow, PyTorch, and JAX, support the @ operator as well. However, you cannot use @ on pure Python arrays (i.e., lists of lists).

{163}------------------------------------------------

The function that we used to generate the data is  $y = 4 + 3x_1 +$  Gaussian noise. Let's see what the equation found:

```
>>> theta best
array([[4.21509616],
      [2.77011339]
```

We would have hoped for  $\theta_0 = 4$  and  $\theta_1 = 3$  instead of  $\theta_0 = 4.215$  and  $\theta_1 = 2.770$ . Close enough, but the noise made it impossible to recover the exact parameters of the original function. The smaller and noisier the dataset, the harder it gets.

Now we can make predictions using  $\hat{\theta}$ :

```
>>> X new = np.array([0], [2])
>>> X new b = add dummy feature(X new) # add x0 = 1 to each instance
\Rightarrow y_predict = X_new_b @ theta_best
>>> y predict
array([[4.21509616],
       [9.75532293]]
```

Let's plot this model's predictions (Figure 4-2):

```
import matplotlib.pyplot as plt
```

```
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")[...] # beautify the figure: add labels, axis, grid, and legend
plt.show()
```

![](img/_page_163_Figure_8.jpeg)

Figure 4-2. Linear regression model predictions

{164}------------------------------------------------

Performing linear regression using Scikit-Learn is relatively straightforward:

```
>>> from sklearn.linear model import LinearRegression
\Rightarrow lin reg = LinearRegression()
\Rightarrow lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new)
array([14.21509616],[9.75532293]]
```

Notice that Scikit-Learn separates the bias term (intercept\_) from the feature weights (coef). The Linear Regression class is based on the scipy. linalg.  $lstsq()$ function (the name stands for "least squares"), which you could call directly:

```
>>> theta best svd, residuals, rank, s = np.linalg.lstsq(X b, y, rcond=1e-6)>>> theta_best_svd
array([[4.21509616],
      [2.77011339]
```

This function computes  $\hat{\theta} = X^+ y$ , where  $X^+$  is the *pseudoinverse* of X (specifically, the Moore-Penrose inverse). You can use np.linalg.pinv() to compute the pseudoinverse directly:

```
>>> np.linalg.pinv(X_b) @ y
array([[4.21509616],[2.77011339]]
```

The pseudoinverse itself is computed using a standard matrix factorization technique called *singular value decomposition* (SVD) that can decompose the training set matrix X into the matrix multiplication of three matrices  $U \Sigma V^{\dagger}$  (see numpy. linalg. svd()). The pseudoinverse is computed as  $X^+ = V\Sigma^+U^{\dagger}$ . To compute the matrix  $\Sigma^+$ , the algorithm takes  $\Sigma$  and sets to zero all values smaller than a tiny threshold value, then it replaces all the nonzero values with their inverse, and finally it transposes the resulting matrix. This approach is more efficient than computing the Normal equation, plus it handles edge cases nicely: indeed, the Normal equation may not work if the matrix  $X^{\dagger}X$  is not invertible (i.e., singular), such as if  $m < n$  or if some features are redundant, but the pseudoinverse is always defined.

### **Computational Complexity**

The Normal equation computes the inverse of  $X^{\dagger} X$ , which is an  $(n + 1) \times (n + 1)$ matrix (where *n* is the number of features). The *computational complexity* of inverting such a matrix is typically about  $O(n^{24})$  to  $O(n^3)$ , depending on the implementation. In other words, if you double the number of features, you multiply the computation time by roughly  $2^{2.4} = 5.3$  to  $2^3 = 8$ .

{165}------------------------------------------------

The SVD approach used by Scikit-Learn's LinearRegression class is about  $O(n^2)$ . If you double the number of features, you multiply the computation time by roughly 4.

![](img/_page_165_Picture_1.jpeg)

Both the Normal equation and the SVD approach get very slow when the number of features grows large (e.g., 100,000). On the positive side, both are linear with regard to the number of instances in the training set (they are  $O(m)$ ), so they handle large training sets efficiently, provided they can fit in memory.

Also, once you have trained your linear regression model (using the Normal equation or any other algorithm), predictions are very fast: the computational complexity is linear with regard to both the number of instances you want to make predictions on and the number of features. In other words, making predictions on twice as many instances (or twice as many features) will take roughly twice as much time.

Now we will look at a very different way to train a linear regression model, which is better suited for cases where there are a large number of features or too many training instances to fit in memory.

### **Gradient Descent**

Gradient descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of gradient descent is to tweak parameters iteratively in order to minimize a cost function.

Suppose you are lost in the mountains in a dense fog, and you can only feel the slope of the ground below your feet. A good strategy to get to the bottom of the valley quickly is to go downhill in the direction of the steepest slope. This is exactly what gradient descent does: it measures the local gradient of the error function with regard to the parameter vector  $\theta$ , and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!

In practice, you start by filling  $\theta$  with random values (this is called *random initialization*). Then you improve it gradually, taking one baby step at a time, each step attempting to decrease the cost function (e.g., the MSE), until the algorithm *converges* to a minimum (see Figure 4-3).

{166}------------------------------------------------

![](img/_page_166_Figure_0.jpeg)

Figure 4-3. In this depiction of gradient descent, the model parameters are initialized randomly and get tweaked repeatedly to minimize the cost function; the learning step size is proportional to the slope of the cost function, so the steps gradually get smaller as the cost approaches the minimum

An important parameter in gradient descent is the size of the steps, determined by the learning rate hyperparameter. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time (see Figure 4-4).

![](img/_page_166_Figure_3.jpeg)

Figure 4-4. Learning rate too small

{167}------------------------------------------------

On the other hand, if the learning rate is too high, you might jump across the valley and end up on the other side, possibly even higher up than you were before. This might make the algorithm diverge, with larger and larger values, failing to find a good solution (see Figure 4-5).

![](img/_page_167_Figure_1.jpeg)

Figure 4-5. Learning rate too high

Additionally, not all cost functions look like nice, regular bowls. There may be holes, ridges, plateaus, and all sorts of irregular terrain, making convergence to the minimum difficult. Figure 4-6 shows the two main challenges with gradient descent. If the random initialization starts the algorithm on the left, then it will converge to a local minimum, which is not as good as the *global minimum*. If it starts on the right, then it will take a very long time to cross the plateau. And if you stop too early, you will never reach the global minimum.

![](img/_page_167_Figure_4.jpeg)

Figure 4-6. Gradient descent pitfalls

{168}------------------------------------------------

Fortunately, the MSE cost function for a linear regression model happens to be a convex function, which means that if you pick any two points on the curve, the line segment joining them is never below the curve. This implies that there are no local minima, just one global minimum. It is also a continuous function with a slope that never changes abruptly.<sup>2</sup> These two facts have a great consequence: gradient descent is guaranteed to approach arbitrarily closely the global minimum (if you wait long enough and if the learning rate is not too high).

While the cost function has the shape of a bowl, it can be an elongated bowl if the features have very different scales. Figure 4-7 shows gradient descent on a training set where features 1 and 2 have the same scale (on the left), and on a training set where feature 1 has much smaller values than feature 2 (on the right).<sup>3</sup>

![](img/_page_168_Figure_2.jpeg)

Figure 4-7. Gradient descent with (left) and without (right) feature scaling

As you can see, on the left the gradient descent algorithm goes straight toward the minimum, thereby reaching it quickly, whereas on the right it first goes in a direction almost orthogonal to the direction of the global minimum, and it ends with a long march down an almost flat valley. It will eventually reach the minimum, but it will take a long time.

![](img/_page_168_Picture_5.jpeg)

When using gradient descent, you should ensure that all features have a similar scale (e.g., using Scikit-Learn's StandardScaler class), or else it will take much longer to converge.

<sup>2</sup> Technically speaking, its derivative is Lipschitz continuous.

<sup>3</sup> Since feature 1 is smaller, it takes a larger change in  $\theta_1$  to affect the cost function, which is why the bowl is elongated along the  $\theta_1$  axis.

{169}------------------------------------------------

This diagram also illustrates the fact that training a model means searching for a combination of model parameters that minimizes a cost function (over the training set). It is a search in the model's *parameter space*. The more parameters a model has, the more dimensions this space has, and the harder the search is: searching for a needle in a 300-dimensional haystack is much trickier than in 3 dimensions. Fortunately, since the cost function is convex in the case of linear regression, the needle is simply at the bottom of the bowl.

#### **Batch Gradient Descent**

To implement gradient descent, you need to compute the gradient of the cost function with regard to each model parameter  $\theta_i$ . In other words, you need to calculate how much the cost function will change if you change  $\theta_i$  just a little bit. This is called a *partial derivative*. It is like asking, "What is the slope of the mountain under my feet if I face east"? and then asking the same question facing north (and so on for all other dimensions, if you can imagine a universe with more than three dimensions). Equation 4-5 computes the partial derivative of the MSE with regard to parameter  $\theta$ <sub>i</sub>, noted  $\partial$  MSE( $\theta$ ) /  $\partial \theta_i$ .

Equation 4-5. Partial derivatives of the cost function

$$
\frac{\partial}{\partial \theta_j} \text{MSE}(\mathbf{\Theta}) = \frac{2}{m} \sum_{i=1}^{m} \left( \mathbf{\Theta}^\mathsf{T} \mathbf{x}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

Instead of computing these partial derivatives individually, you can use Equation 4-6 to compute them all in one go. The gradient vector, noted  $\nabla_{\theta}MSE(\theta)$ , contains all the partial derivatives of the cost function (one for each model parameter).

Equation 4-6. Gradient vector of the cost function

$$
\nabla_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) = \begin{pmatrix} \frac{\partial}{\partial \theta_0} \text{MSE}(\boldsymbol{\theta}) \\ \frac{\partial}{\partial \theta_1} \text{MSE}(\boldsymbol{\theta}) \\ \vdots \\ \frac{\partial}{\partial \theta_n} \text{MSE}(\boldsymbol{\theta}) \end{pmatrix} = \frac{2}{m} \mathbf{X}^{\mathsf{T}} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})
$$

{170}------------------------------------------------

![](img/_page_170_Picture_0.jpeg)

Notice that this formula involves calculations over the full training set  $X$ , at each gradient descent step! This is why the algorithm is called batch gradient descent: it uses the whole batch of training data at every step (actually, full gradient descent would probably be a better name). As a result, it is terribly slow on very large training sets (we will look at some much faster gradient descent algorithms shortly). However, gradient descent scales well with the number of features; training a linear regression model when there are hundreds of thousands of features is much faster using gradient descent than using the Normal equation or SVD decomposition.

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill. This means subtracting  $\nabla_{\theta}MSE(\theta)$  from  $\theta$ . This is where the learning rate  $\eta$  comes into play:<sup>4</sup> multiply the gradient vector by  $\eta$  to determine the size of the downhill step (Equation 4-7).

Equation 4-7. Gradient descent step

```
\theta^{(next\,step)} = \theta - \eta \nabla_{\theta} \text{MSE}(\theta)
```

Let's look at a quick implementation of this algorithm:

```
eta = 0.1 # learning rate
n epochs = 1000
m = len(X_b) # number of instances
np.random.seed(42)
theta = np.random.randn(2, 1) # randomly initialized model parametersfor epoch in range(n_epochs):
    gradients = 2 / m * X b.T @ (X b @ theta - y)theta = theta - eta * gradients
```

That wasn't too hard! Each iteration over the training set is called an epoch. Let's look at the resulting theta:

```
>>> theta
array([[4.21509616],
      [2.77011339]
```

Hey, that's exactly what the Normal equation found! Gradient descent worked perfectly. But what if you had used a different learning rate (eta)? Figure 4-8 shows the first 20 steps of gradient descent using three different learning rates. The line at the bottom of each plot represents the random starting point, then each epoch is represented by a darker and darker line.

<sup>4</sup> Eta  $(n)$  is the seventh letter of the Greek alphabet.

{171}------------------------------------------------

![](img/_page_171_Figure_0.jpeg)

Figure 4-8. Gradient descent with various learning rates

On the left, the learning rate is too low: the algorithm will eventually reach the solution, but it will take a long time. In the middle, the learning rate looks pretty good: in just a few epochs, it has already converged to the solution. On the right, the learning rate is too high: the algorithm diverges, jumping all over the place and actually getting further and further away from the solution at every step.

To find a good learning rate, you can use grid search (see Chapter 2). However, you may want to limit the number of epochs so that grid search can eliminate models that take too long to converge.

You may wonder how to set the number of epochs. If it is too low, you will still be far away from the optimal solution when the algorithm stops; but if it is too high, you will waste time while the model parameters do not change anymore. A simple solution is to set a very large number of epochs but to interrupt the algorithm when the gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny number  $\epsilon$  (called the *tolerance*)—because this happens when gradient descent has (almost) reached the minimum.

#### **Convergence Rate**

When the cost function is convex and its slope does not change abruptly (as is the case for the MSE cost function), batch gradient descent with a fixed learning rate will eventually converge to the optimal solution, but you may have to wait a while: it can take  $O(1/\epsilon)$  iterations to reach the optimum within a range of  $\epsilon$ , depending on the shape of the cost function. If you divide the tolerance by 10 to have a more precise solution, then the algorithm may have to run about 10 times longer.

{172}------------------------------------------------

### **Stochastic Gradient Descent**

The main problem with batch gradient descent is the fact that it uses the whole training set to compute the gradients at every step, which makes it very slow when the training set is large. At the opposite extreme, *stochastic gradient descent* picks a random instance in the training set at every step and computes the gradients based only on that single instance. Obviously, working on a single instance at a time makes the algorithm much faster because it has very little data to manipulate at every iteration. It also makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration (stochastic GD can be implemented as an out-of-core algorithm; see Chapter 1).

On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much less regular than batch gradient descent: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, never settling down (see Figure 4-9). Once the algorithm stops, the final parameter values will be good, but not optimal.

![](img/_page_172_Figure_3.jpeg)

Figure 4-9. With stochastic gradient descent, each training step is much faster but also much more stochastic than when using batch gradient descent

When the cost function is very irregular (as in Figure 4-6), this can actually help the algorithm jump out of local minima, so stochastic gradient descent has a better chance of finding the global minimum than batch gradient descent does.

Therefore, randomness is good to escape from local optima, but bad because it means that the algorithm can never settle at the minimum. One solution to this dilemma is to gradually reduce the learning rate. The steps start out large (which helps make quick progress and escape local minima), then get smaller and smaller, allowing the algorithm to settle at the global minimum. This process is akin to simulated annealing, an algorithm inspired by the process in metallurgy of annealing, where molten metal is slowly cooled down. The function that determines the learning rate

{173}------------------------------------------------

at each iteration is called the *learning schedule*. If the learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up frozen halfway to the minimum. If the learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early.

This code implements stochastic gradient descent using a simple learning schedule:

```
n epochs = 50t0, t1 = 5, 50 # learning schedule hyperparameters
def learning schedule(t):
    return t0 / (t + t1)np.random.seed(42)
theta = np.random.randn(2, 1) # random initializationfor epoch in range(n_epochs):
    for iteration in range(m):
       random index = np.random.randnint(m)xi = X_b[random_index : random_index + 1]yi = y[random index : random index + 1]gradients = 2 * x i . T @ (xi @ theta - yi) # for SGD, do not divide by meta = learning\_schedule(epoch * m + iteration)theta = theta - eta * gradients
```

By convention we iterate by rounds of  $m$  iterations; each round is called an *epoch*, as earlier. While the batch gradient descent code iterated 1,000 times through the whole training set, this code goes through the training set only 50 times and reaches a pretty good solution:

>>> theta array([[4.21076011],  $[2.74856079]$ 

Figure 4-10 shows the first 20 steps of training (notice how irregular the steps are).

Note that since instances are picked randomly, some instances may be picked several times per epoch, while others may not be picked at all. If you want to be sure that the algorithm goes through every instance at each epoch, another approach is to shuffle the training set (making sure to shuffle the input features and the labels jointly), then go through it instance by instance, then shuffle it again, and so on. However, this approach is more complex, and it generally does not improve the result.

{174}------------------------------------------------

![](img/_page_174_Figure_0.jpeg)

Figure 4-10. The first 20 steps of stochastic gradient descent

![](img/_page_174_Picture_2.jpeg)

When using stochastic gradient descent, the training instances must be independent and identically distributed (IID) to ensure that the parameters get pulled toward the global optimum, on average. A simple way to ensure this is to shuffle the instances during training (e.g., pick each instance randomly, or shuffle the training set at the beginning of each epoch). If you do not shuffle the instances—for example, if the instances are sorted by label then SGD will start by optimizing for one label, then the next, and so on, and it will not settle close to the global minimum.

To perform linear regression using stochastic GD with Scikit-Learn, you can use the SGDRegressor class, which defaults to optimizing the MSE cost function. The following code runs for maximum 1,000 epochs (max iter) or until the loss drops by less than  $10^{-5}$  (tol) during 100 epochs (n\_iter\_no\_change). It starts with a learning rate of 0.01 (eta0), using the default learning schedule (different from the one we used). Lastly, it does not use any regularization (penalty=None; more details on this shortly):

```
from sklearn.linear_model import SGDRegressor
sqd req = SGDReqressor(max iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n iter no change=100, random state=42)
sgd_reg.fit(X, y.ravel()) # y.ravel() because fit() expects 1D targets
```

{175}------------------------------------------------

Once again, you find a solution quite close to the one returned by the Normal equation:

```
>>> sgd_reg.intercept_, sgd_reg.coef_
(array([4.21278812]), array([2.77270267]))
```

![](img/_page_175_Picture_2.jpeg)

All Scikit-Learn estimators can be trained using the fit() method, but some estimators also have a partial fit() method that you can call to run a single round of training on one or more instances (it ignores hyperparameters like max\_iter or tol). Repeatedly calling partial\_fit() will gradually train the model. This is useful when you need more control over the training process. Other models have a warm\_start hyperparameter instead (and some have both): if you set warm\_start=True, calling the fit() method on a trained model will not reset the model; it will just continue training where it left off, respecting hyperparameters like max\_iter and tol. Note that fit() resets the iteration counter used by the learning schedule, while partial fit() does not.

### **Mini-Batch Gradient Descent**

The last gradient descent algorithm we will look at is called mini-batch gradient *descent*. It is straightforward once you know batch and stochastic gradient descent: at each step, instead of computing the gradients based on the full training set (as in batch GD) or based on just one instance (as in stochastic GD), mini-batch GD computes the gradients on small random sets of instances called *mini-batches*. The main advantage of mini-batch GD over stochastic GD is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.

The algorithm's progress in parameter space is less erratic than with stochastic GD, especially with fairly large mini-batches. As a result, mini-batch GD will end up walking around a bit closer to the minimum than stochastic GD-but it may be harder for it to escape from local minima (in the case of problems that suffer from local minima, unlike linear regression with the MSE cost function). Figure 4-11 shows the paths taken by the three gradient descent algorithms in parameter space during training. They all end up near the minimum, but batch GD's path actually stops at the minimum, while both stochastic GD and mini-batch GD continue to walk around. However, don't forget that batch GD takes a lot of time to take each step, and stochastic GD and mini-batch GD would also reach the minimum if you used a good learning schedule.

{176}------------------------------------------------

![](img/_page_176_Figure_0.jpeg)

Figure 4-11. Gradient descent paths in parameter space

Table 4-1 compares the algorithms we've discussed so far for linear regression<sup>5</sup> (recall that  $m$  is the number of training instances and  $n$  is the number of features).

| Algorithm       |      | Large m Out-of-core support Large n Hyperparams Scaling required Scikit-Learn |      |          |     |                  |
|-----------------|------|-------------------------------------------------------------------------------|------|----------|-----|------------------|
| Normal equation | Fast | No                                                                            | Slow | 0        | No  | N/A              |
| SVD             | Fast | No                                                                            | Slow | 0        | No  | LinearRegression |
| Batch GD        | Slow | No                                                                            | Fast |          | Yes | N/A              |
| Stochastic GD   | Fast | Yes                                                                           | Fast | >)       | Yes | SGDRegressor     |
| Mini-batch GD   | Fast | Yes                                                                           | Fast | $\geq$ 2 | Yes | N/A              |

Table 4-1. Comparison of algorithms for linear regression

There is almost no difference after training: all these algorithms end up with very similar models and make predictions in exactly the same way.

### **Polynomial Regression**

What if your data is more complex than a straight line? Surprisingly, you can use a linear model to fit nonlinear data. A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called polynomial regression.

<sup>5</sup> While the Normal equation can only perform linear regression, the gradient descent algorithms can be used to train many other models, as you'll see.

{177}------------------------------------------------

Let's look at an example. First, we'll generate some nonlinear data (see Figure 4-12), based on a simple *quadratic equation*—that's an equation of the form  $y = ax^2 + bx +$  $c$ -plus some noise:

![](img/_page_177_Figure_1.jpeg)

Figure 4-12. Generated nonlinear and noisy dataset

Clearly, a straight line will never fit this data properly. So let's use Scikit-Learn's PolynomialFeatures class to transform our training data, adding the square (seconddegree polynomial) of each feature in the training set as a new feature (in this case there is just one feature):

```
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly features = PolynomialFeatures(degree=2, include bias=False)
\Rightarrow X_poly = poly_features.fit_transform(X)
\Rightarrow \times [0]
array([-0.75275929])
\Rightarrow X poly[0]
array([ -0.75275929, 0.56664654])
```

X\_poly now contains the original feature of X plus the square of this feature. Now we can fit a LinearRegression model to this extended training data (Figure 4-13):

```
\Rightarrow lin reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([1.78134581]), array([0.93366893, 0.56456263]]))
```

{178}------------------------------------------------

![](img/_page_178_Figure_0.jpeg)

Figure 4-13. Polynomial regression model predictions

Not bad: the model estimates  $\hat{y} = 0.56x_1^2 + 0.93x_1 + 1.78$  when in fact the original function was  $y = 0.5x_1^2 + 1.0x_1 + 2.0$  + Gaussian noise.

Note that when there are multiple features, polynomial regression is capable of finding relationships between features, which is something a plain linear regression model cannot do. This is made possible by the fact that PolynomialFeatures also adds all combinations of features up to the given degree. For example, if there were two features a and b, PolynomialFeatures with degree=3 would not only add the features  $a^2$ ,  $a^3$ ,  $b^2$ , and  $b^3$ , but also the combinations ab,  $a^2b$ , and ab<sup>2</sup>.

![](img/_page_178_Picture_4.jpeg)

PolynomialFeatures( $degree=d$ ) transforms an array containing  $n$ features into an array containing  $(n + d)! / d!n!$  features, where n! is the *factorial* of *n*, equal to  $1 \times 2 \times 3 \times \cdots \times n$ . Beware of the combinatorial explosion of the number of features!

### **Learning Curves**

If you perform high-degree polynomial regression, you will likely fit the training data much better than with plain linear regression. For example, Figure 4-14 applies a 300-degree polynomial model to the preceding training data, and compares the result with a pure linear model and a quadratic model (second-degree polynomial). Notice how the 300-degree polynomial model wiggles around to get as close as possible to the training instances.

{179}------------------------------------------------

![](img/_page_179_Figure_0.jpeg)

Figure 4-14. High-degree polynomial regression

This high-degree polynomial regression model is severely overfitting the training data, while the linear model is underfitting it. The model that will generalize best in this case is the quadratic model, which makes sense because the data was generated using a quadratic model. But in general you won't know what function generated the data, so how can you decide how complex your model should be? How can you tell that your model is overfitting or underfitting the data?

In Chapter 2 you used cross-validation to get an estimate of a model's generalization performance. If a model performs well on the training data but generalizes poorly according to the cross-validation metrics, then your model is overfitting. If it performs poorly on both, then it is underfitting. This is one way to tell when a model is too simple or too complex.

Another way to tell is to look at the *learning curves*, which are plots of the model's training error and validation error as a function of the training iteration: just evaluate the model at regular intervals during training on both the training set and the validation set, and plot the results. If the model cannot be trained incrementally (i.e., if it does not support partial fit() or warm start), then you must train it several times on gradually larger subsets of the training set.

Scikit-Learn has a useful learning curve() function to help with this: it trains and evaluates the model using cross-validation. By default it retrains the model on growing subsets of the training set, but if the model supports incremental learning you can set exploit\_incremental\_learning=True when calling learning\_curve() and it will train the model incrementally instead. The function returns the training set sizes at which it evaluated the model, and the training and validation scores it

{180}------------------------------------------------

measured for each size and for each cross-validation fold. Let's use this function to look at the learning curves of the plain linear regression model (see Figure 4-15):

![](img/_page_180_Figure_1.jpeg)

Figure 4-15. Learning curves

 $0.0$ 

 $10$ 

 $20$ 

 $30$ 

 $40$ 

Training set size

 $50$ 

60

 $70$ 

80

This model is underfitting. To see why, first let's look at the training error. When there are just one or two instances in the training set, the model can fit them perfectly, which is why the curve starts at zero. But as new instances are added to the training set, it becomes impossible for the model to fit the training data perfectly, both because the data is noisy and because it is not linear at all. So the error on the training data goes up until it reaches a plateau, at which point adding new instances to the training set doesn't make the average error much better or worse. Now let's look at the validation error. When the model is trained on very few training instances, it is incapable of generalizing properly, which is why the validation error is initially quite large. Then, as the model is shown more training examples, it learns, and thus the validation error slowly goes down. However, once again a straight line cannot do a good job of modeling the data, so the error ends up at a plateau, very close to the other curve.

{181}------------------------------------------------

These learning curves are typical of a model that's underfitting. Both curves have reached a plateau; they are close and fairly high.

![](img/_page_181_Picture_1.jpeg)

If your model is underfitting the training data, adding more training examples will not help. You need to use a better model or come up with better features.

Now let's look at the learning curves of a 10th-degree polynomial model on the same data (Figure 4-16):

```
from sklearn.pipeline import make pipeline
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())
train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
[...] # same as earlier
```

![](img/_page_181_Figure_5.jpeg)

Figure 4-16. Learning curves for the 10th-degree polynomial model

These learning curves look a bit like the previous ones, but there are two very important differences:

- The error on the training data is much lower than before.
- There is a gap between the curves. This means that the model performs significantly better on the training data than on the validation data, which is the

{182}------------------------------------------------

hallmark of an overfitting model. If you used a much larger training set, however, the two curves would continue to get closer.

![](img/_page_182_Picture_1.jpeg)

One way to improve an overfitting model is to feed it more training data until the validation error reaches the training error.

### The Bias/Variance Trade-Off

An important theoretical result of statistics and machine learning is the fact that a model's generalization error can be expressed as the sum of three very different errors<sup>.</sup>

**Bias** 

This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.<sup>6</sup>

Variance

This part is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance and thus overfit the training data.

Irreducible error

This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).

Increasing a model's complexity will typically increase its variance and reduce its bias. Conversely, reducing a model's complexity increases its bias and reduces its variance. This is why it is called a trade-off.

### **Regularized Linear Models**

As you saw in Chapters 1 and 2, a good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. A simple way to regularize a polynomial model is to reduce the number of polynomial degrees.

<sup>6</sup> This notion of bias is not to be confused with the bias term of linear models.

{183}------------------------------------------------

For a linear model, regularization is typically achieved by constraining the weights of the model. We will now look at ridge regression, lasso regression, and elastic net regression, which implement three different ways to constrain the weights.

#### **Ridge Regression**

Ridge regression (also called Tikhonov regularization) is a regularized version of linear regression: a *regularization term* equal to  $\frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2$  is added to the MSE. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. Note that the regularization term should only be added to the cost function during training. Once the model is trained, you want to use the unregularized MSE (or the RMSE) to evaluate the model's performance.

The hyperparameter  $\alpha$  controls how much you want to regularize the model. If  $\alpha$  = 0, then ridge regression is just linear regression. If  $\alpha$  is very large, then all weights end up very close to zero and the result is a flat line going through the data's mean. Equation 4-8 presents the ridge regression cost function.<sup>7</sup>

Equation 4-8. Ridge regression cost function

 $J(\mathbf{\Theta}) = \text{MSE}(\mathbf{\Theta}) + \frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2$ 

Note that the bias term  $\theta_0$  is not regularized (the sum starts at  $i = 1$ , not 0). If we define w as the vector of feature weights ( $\theta_1$  to  $\theta_n$ ), then the regularization term is equal to  $\alpha(\|\mathbf{w}\|_{2})^2$  / m, where  $\|\mathbf{w}\|_{2}$  represents the  $\ell_2$  norm of the weight vector.<sup>8</sup> For batch gradient descent, just add  $2\alpha w / m$  to the part of the MSE gradient vector that corresponds to the feature weights, without adding anything to the gradient of the bias term (see Equation  $4-6$ ).

![](img/_page_183_Picture_7.jpeg)

It is important to scale the data (e.g., using a StandardScaler) before performing ridge regression, as it is sensitive to the scale of the input features. This is true of most regularized models.

Figure 4-17 shows several ridge models that were trained on some very noisy linear data using different  $\alpha$  values. On the left, plain ridge models are used, leading to linear predictions. On the right, the data is first expanded using

<sup>7</sup> It is common to use the notation  $J(\theta)$  for cost functions that don't have a short name; I'll often use this notation throughout the rest of this book. The context will make it clear which cost function is being discussed.

<sup>8</sup> Norms are discussed in Chapter 2.

{184}------------------------------------------------

PolynomialFeatures(degree=10), then it is scaled using a StandardScaler, and finally the ridge models are applied to the resulting features: this is polynomial regression with ridge regularization. Note how increasing  $\alpha$  leads to flatter (i.e., less extreme, more reasonable) predictions, thus reducing the model's variance but increasing its bias.

![](img/_page_184_Figure_1.jpeg)

Figure 4-17. Linear (left) and a polynomial (right) models, both with various levels of ridge regularization

As with linear regression, we can perform ridge regression either by computing a closed-form equation or by performing gradient descent. The pros and cons are the same. Equation 4-9 shows the closed-form solution, where **A** is the  $(n + 1) \times (n + 1)$ *identity matrix*,<sup>9</sup> except with a 0 in the top-left cell, corresponding to the bias term.

Equation 4-9. Ridge regression closed-form solution

 $\widehat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{A})^{-1} \mathbf{X}^\top \mathbf{y}$ 

Here is how to perform ridge regression with Scikit-Learn using a closed-form solution (a variant of Equation 4-9 that uses a matrix factorization technique by André-Louis Cholesky):

```
>>> from sklearn.linear_model import Ridge
>>> ridge_reg = Ridge(alpha=0.1, solver="cholesky")
>>> ridge_reg.fit(X, y)
>>> ridge_reg.predict([[1.5]])
array([[1.55325833]])
```

<sup>9</sup> A square matrix full of 0s except for 1s on the main diagonal (top left to bottom right).

{185}------------------------------------------------

And using stochastic gradient descent:<sup>10</sup>

```
>>> sgd reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                           max iter=1000, eta0=0.01, random state=42)
\ddotsc\ddotsc>>> sgd reg.fit(X, y.ravel()) # y.ravel() because fit() expects 1D targets
\gg sgd_reg.predict([1.5]])
array([1.55302613])
```

The penalty hyperparameter sets the type of regularization term to use. Specifying "12" indicates that you want SGD to add a regularization term to the MSE cost function equal to alpha times the square of the  $\ell_2$  norm of the weight vector. This is just like ridge regression, except there's no division by  $m$  in this case; that's why we passed alpha=0.1 / m, to get the same result as Ridge(alpha=0.1).

![](img/_page_185_Picture_3.jpeg)

The RidgeCV class also performs ridge regression, but it automatically tunes hyperparameters using cross-validation. It's roughly equivalent to using GridSearchCV, but it's optimized for ridge regression and runs much faster. Several other estimators (mostly linear) also have efficient CV variants, such as LassoCV and FlasticNetCV.

### **Lasso Regression**

Least absolute shrinkage and selection operator regression (usually simply called lasso regression) is another regularized version of linear regression: just like ridge regression, it adds a regularization term to the cost function, but it uses the  $\ell_1$  norm of the weight vector instead of the square of the  $\ell_2$  norm (see Equation 4-10). Notice that the  $\ell_1$  norm is multiplied by 2 $\alpha$ , whereas the  $\ell_2$  norm was multiplied by  $\alpha / m$ in ridge regression. These factors were chosen to ensure that the optimal  $\alpha$  value is independent from the training set size: different norms lead to different factors (see Scikit-Learn issue #15657 for more details).

Equation 4-10. Lasso regression cost function

 $J(\mathbf{\Theta}) = \text{MSE}(\mathbf{\Theta}) + 2\alpha \sum_{i=1}^{n} |\theta_i|$ 

Figure 4-18 shows the same thing as Figure 4-17 but replaces the ridge models with lasso models and uses different  $\alpha$  values

<sup>10</sup> Alternatively, you can use the Ridge class with the "sag" solver. Stochastic average GD is a variant of stochastic GD. For more details, see the presentation "Minimizing Finite Sums with the Stochastic Average Gradient Algorithm" by Mark Schmidt et al. from the University of British Columbia.

{186}------------------------------------------------

![](img/_page_186_Figure_0.jpeg)

Figure 4-18. Linear (left) and polynomial (right) models, both using various levels of lasso regularization

An important characteristic of lasso regression is that it tends to eliminate the weights of the least important features (i.e., set them to zero). For example, the dashed line in the righthand plot in Figure 4-18 (with  $\alpha = 0.01$ ) looks roughly cubic: all the weights for the high-degree polynomial features are equal to zero. In other words, lasso regression automatically performs feature selection and outputs a sparse model with few nonzero feature weights.

You can get a sense of why this is the case by looking at Figure 4-19: the axes represent two model parameters, and the background contours represent different loss functions. In the top-left plot, the contours represent the  $\ell_1$  loss  $(|\theta_1| + |\theta_2|)$ , which drops linearly as you get closer to any axis. For example, if you initialize the model parameters to  $\theta_1 = 2$  and  $\theta_2 = 0.5$ , running gradient descent will decrement both parameters equally (as represented by the dashed yellow line); therefore  $\theta$ , will reach 0 first (since it was closer to 0 to begin with). After that, gradient descent will roll down the gutter until it reaches  $\theta_1 = 0$  (with a bit of bouncing around, since the gradients of  $\ell_1$  never get close to 0: they are either -1 or 1 for each parameter). In the top-right plot, the contours represent lasso regression's cost function (i.e., an MSE cost function plus an  $\ell_1$  loss). The small white circles show the path that gradient descent takes to optimize some model parameters that were initialized around  $\theta_1$  = 0.25 and  $\theta_2$  = -1: notice once again how the path quickly reaches  $\theta_2$  = 0, then rolls down the gutter and ends up bouncing around the global optimum (represented by the red square). If we increased  $\alpha$ , the global optimum would move left along the dashed yellow line, while if we decreased  $\alpha$ , the global optimum would move right (in this example, the optimal parameters for the unregularized MSE are  $\theta_1 = 2$  and  $\theta_2 =$  $(0.5)$ .

{187}------------------------------------------------

![](img/_page_187_Figure_0.jpeg)

Figure 4-19. Lasso versus ridge regularization

The two bottom plots show the same thing but with an  $\ell_2$  penalty instead. In the bottom-left plot, you can see that the  $\ell_2$  loss decreases as we get closer to the origin, so gradient descent just takes a straight path toward that point. In the bottom-right plot, the contours represent ridge regression's cost function (i.e., an MSE cost function plus an  $\ell$ , loss). As you can see, the gradients get smaller as the parameters approach the global optimum, so gradient descent naturally slows down. This limits the bouncing around, which helps ridge converge faster than lasso regression. Also note that the optimal parameters (represented by the red square) get closer and closer to the origin when you increase  $\alpha$ , but they never get eliminated entirely.

![](img/_page_187_Picture_3.jpeg)

To keep gradient descent from bouncing around the optimum at the end when using lasso regression, you need to gradually reduce the learning rate during training. It will still bounce around the optimum, but the steps will get smaller and smaller, so it will converge.

{188}------------------------------------------------

The lasso cost function is not differentiable at  $\theta_i = 0$  (for  $i = 1, 2, \dots, n$ ), but gradient descent still works if you use a *subgradient vector*  $\mathbf{g}^{11}$  instead when any  $\theta_i = 0$ . Equation 4-11 shows a subgradient vector equation you can use for gradient descent with the lasso cost function.

Equation 4-11. Lasso regression subgradient vector

$$
g(\mathbf{\theta}, J) = \nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) + 2\alpha \begin{pmatrix} \text{sign } (\theta_1) \\ \text{sign } (\theta_2) \\ \vdots \\ \text{sign } (\theta_n) \end{pmatrix} \quad \text{where } \text{sign } (\theta_i) = \begin{cases} -1 & \text{if } \theta_i < 0 \\ 0 & \text{if } \theta_i = 0 \\ +1 & \text{if } \theta_i > 0 \end{cases}
$$

 $\lambda$  and  $\lambda$ 

Here is a small Scikit-Learn example using the Lasso class:

```
>>> from sklearn.linear_model import Lasso
\Rightarrow lasso reg = Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
\gg lasso_reg.predict([1.5])
array([1.53788174])
```

Note that you could instead use SGDRegressor (penalty="l1", alpha=0.1).

### **Elastic Net Regression**

*Elastic net regression* is a middle ground between ridge regression and lasso regression. The regularization term is a weighted sum of both ridge and lasso's regularization terms, and you can control the mix ratio  $r$ . When  $r = 0$ , elastic net is equivalent to ridge regression, and when  $r = 1$ , it is equivalent to lasso regression (Equation  $4-12$ ).

Equation 4-12. Elastic net cost function

 $J(\mathbf{\Theta}) = \text{MSE}(\mathbf{\Theta}) + r(2\alpha \Sigma_{i=1}^n |\theta_i|) + (1-r)(\frac{\alpha}{m} \Sigma_{i=1}^n \theta_i^2)$ 

So when should you use elastic net regression, or ridge, lasso, or plain linear regression (i.e., without any regularization)? It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain linear regression. Ridge is a good default, but if you suspect that only a few features are useful, you should prefer lasso or elastic net because they tend to reduce the useless features' weights down to zero, as discussed earlier. In general, elastic net is preferred over

<sup>11</sup> You can think of a subgradient vector at a nondifferentiable point as an intermediate vector between the gradient vectors around that point.

{189}------------------------------------------------

lasso because lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.

Here is a short example that uses Scikit-Learn's ElasticNet (11 ratio corresponds to the mix ratio  $r$ ):

```
>>> from sklearn.linear_model import ElasticNet
\gg elastic net = ElasticNet(alpha=0.1, l1 ratio=0.5)
>>> elastic_net.fit(X, y)
>>> elastic_net.predict([[1.5]])
array([1.54333232])
```

### **Early Stopping**

A very different way to regularize iterative learning algorithms such as gradient descent is to stop training as soon as the validation error reaches a minimum. This is called early stopping. Figure 4-20 shows a complex model (in this case, a high-degree polynomial regression model) being trained with batch gradient descent on the quadratic dataset we used earlier. As the epochs go by, the algorithm learns, and its prediction error (RMSE) on the training set goes down, along with its prediction error on the validation set. After a while, though, the validation error stops decreasing and starts to go back up. This indicates that the model has started to overfit the training data. With early stopping you just stop training as soon as the validation error reaches the minimum. It is such a simple and efficient regularization technique that Geoffrey Hinton called it a "beautiful free lunch".

![](img/_page_189_Figure_5.jpeg)

Figure 4-20. Early stopping regularization

{190}------------------------------------------------

![](img/_page_190_Picture_0.jpeg)

With stochastic and mini-batch gradient descent, the curves are not so smooth, and it may be hard to know whether you have reached the minimum or not. One solution is to stop only after the validation error has been above the minimum for some time (when you are confident that the model will not do any better), then roll back the model parameters to the point where the validation error was at a minimum.

Here is a basic implementation of early stopping:

```
from copy import deepcopy
from sklearn.metrics import mean squared error
from sklearn.preprocessing import StandardScaler
X_train, y_train, X_valid, y_valid = [...] # split the quadratic dataset
preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_value_prep = preprocessing.trainSform(X_value)sgd reg = SGDRegression(penalty=None, eta0=0.002, random state=42)n epochs = 500
best valid rmse = float('inf')for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
   val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)
    if val_error < best_valid_rmse:
       best valid rme = val error
       best_model = deepcopy(sgd_reg)
```

This code first adds the polynomial features and scales all the input features, both for the training set and for the validation set (the code assumes that you have split the original training set into a smaller training set and a validation set). Then it creates an SGDRegressor model with no regularization and a small learning rate. In the training loop, it calls partial fit() instead of fit(), to perform incremental learning. At each epoch, it measures the RMSE on the validation set. If it is lower than the lowest RMSE seen so far, it saves a copy of the model in the best\_model variable. This implementation does not actually stop training, but it lets you revert to the best model after training. Note that the model is copied using copy.deepcopy(), because it copies both the model's hyperparameters *and* the learned parameters. In contrast, sklearn.base.clone() only copies the model's hyperparameters.

{191}------------------------------------------------

### **Logistic Regression**

As discussed in Chapter 1, some regression algorithms can be used for classification (and vice versa). Logistic regression (also called *logit regression*) is commonly used to estimate the probability that an instance belongs to a particular class (e.g., what is the probability that this email is spam?). If the estimated probability is greater than a given threshold (typically 50%), then the model predicts that the instance belongs to that class (called the *positive class*, labeled "1"), and otherwise it predicts that it does not (i.e., it belongs to the *negative class*, labeled "0"). This makes it a binary classifier.

### **Estimating Probabilities**

So how does logistic regression work? Just like a linear regression model, a logistic regression model computes a weighted sum of the input features (plus a bias term), but instead of outputting the result directly like the linear regression model does, it outputs the *logistic* of this result (see Equation 4-13).

Equation 4-13. Logistic regression model estimated probability (vectorized form)

 $\hat{p} = h_{\mathbf{\theta}}(\mathbf{x}) = \sigma(\mathbf{\theta}^{\mathsf{T}}\mathbf{x})$ 

The logistic—noted  $\sigma(\cdot)$ —is a *sigmoid function* (i.e., S-shaped) that outputs a number between 0 and 1. It is defined as shown in Equation 4-14 and Figure 4-21.

Equation 4-14. Logistic function

$$
\sigma(t) = \frac{1}{1 + \exp(-t)}
$$

![](img/_page_191_Figure_9.jpeg)

Figure 4-21. Logistic function

{192}------------------------------------------------

Once the logistic regression model has estimated the probability  $\hat{p} = h_0(\mathbf{x})$  that an instance x belongs to the positive class, it can make its prediction  $\hat{y}$  easily (see Equation  $4-15$ ).

Equation 4-15. Logistic regression model prediction using a 50% threshold probability

 $\hat{y} = \begin{cases} 0 & \text{if } \hat{p} < 0.5 \\ 1 & \text{if } \hat{p} \ge 0.5 \end{cases}$ 

Notice that  $\sigma(t) < 0.5$  when  $t < 0$ , and  $\sigma(t) \ge 0.5$  when  $t \ge 0$ , so a logistic regression model using the default threshold of 50% probability predicts 1 if  $\theta^{\dagger}$  x is positive and 0 if it is negative.

![](img/_page_192_Picture_4.jpeg)

The score  $t$  is often called the *logit*. The name comes from the fact that the logit function, defined as  $logit(p) = log(p / (1 - p))$ , is the inverse of the logistic function. Indeed, if you compute the logit of the estimated probability  $p$ , you will find that the result is t. The logit is also called the *log-odds*, since it is the log of the ratio between the estimated probability for the positive class and the estimated probability for the negative class.

#### **Training and Cost Function**

Now you know how a logistic regression model estimates probabilities and makes predictions. But how is it trained? The objective of training is to set the parameter vector  $\theta$  so that the model estimates high probabilities for positive instances ( $y = 1$ ) and low probabilities for negative instances ( $y = 0$ ). This idea is captured by the cost function shown in Equation 4-16 for a single training instance x.

Equation 4-16. Cost function of a single training instance

$$
c(\mathbf{\Theta}) = \begin{cases} -\log(\widehat{p}) & \text{if } y = 1\\ -\log(1-\widehat{p}) & \text{if } y = 0 \end{cases}
$$

This cost function makes sense because  $-\log(t)$  grows very large when t approaches 0, so the cost will be large if the model estimates a probability close to 0 for a positive instance, and it will also be large if the model estimates a probability close to 1 for a negative instance. On the other hand,  $-\log(t)$  is close to 0 when t is close to 1, so the cost will be close to 0 if the estimated probability is close to 0 for a negative instance or close to 1 for a positive instance, which is precisely what we want.

{193}------------------------------------------------

The cost function over the whole training set is the average cost over all training instances. It can be written in a single expression called the log loss, shown in Equation 4-17.

Equation 4-17. Logistic regression cost function (log loss)

 $J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} log(\hat{p}^{(i)}) + (1 - y^{(i)}) log(1 - \hat{p}^{(i)}) \right]$ 

![](img/_page_193_Picture_3.jpeg)

The log loss was not just pulled out of a hat. It can be shown mathematically (using Bayesian inference) that minimizing this loss will result in the model with the maximum likelihood of being optimal, assuming that the instances follow a Gaussian distribution around the mean of their class. When you use the log loss, this is the implicit assumption you are making. The more wrong this assumption is, the more biased the model will be. Similarly, when we used the MSE to train linear regression models, we were implicitly assuming that the data was purely linear, plus some Gaussian noise. So, if the data is not linear (e.g., if it's quadratic) or if the noise is not Gaussian (e.g., if outliers are not exponentially rare), then the model will be biased.

The bad news is that there is no known closed-form equation to compute the value of  $\theta$  that minimizes this cost function (there is no equivalent of the Normal equation). But the good news is that this cost function is convex, so gradient descent (or any other optimization algorithm) is guaranteed to find the global minimum (if the learning rate is not too large and you wait long enough). The partial derivatives of the cost function with regard to the  $j<sup>th</sup>$  model parameter  $\theta_i$  are given by Equation 4-18.

Equation 4-18. Logistic cost function partial derivatives

$$
\frac{\partial}{\partial \theta_j} \mathbf{J}(\mathbf{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \sigma \big(\mathbf{\Theta}^\mathsf{T} \mathbf{x}^{(i)}\big) - y^{(i)}\right) x_j^{(i)}
$$

This equation looks very much like Equation 4-5: for each instance it computes the prediction error and multiplies it by the  $j<sup>th</sup>$  feature value, and then it computes the average over all training instances. Once you have the gradient vector containing all the partial derivatives, you can use it in the batch gradient descent algorithm. That's it: you now know how to train a logistic regression model. For stochastic GD you would take one instance at a time, and for mini-batch GD you would use a mini-batch at a time.

{194}------------------------------------------------

### **Decision Boundaries**

We can use the iris dataset to illustrate logistic regression. This is a famous dataset that contains the sepal and petal length and width of 150 iris flowers of three different species: Iris setosa, Iris versicolor, and Iris virginica (see Figure 4-22).

![](img/_page_194_Picture_2.jpeg)

Figure 4-22. Flowers of three iris plant species<sup>12</sup>

Let's try to build a classifier to detect the *Iris virginica* type based only on the petal width feature. The first step is to load the data and take a quick peek:

```
>>> from sklearn.datasets import load iris
>>> iris = load_iris(as_frame=True)
>>> list(iris)
['data', 'target', 'frame', 'target names', 'DESCR', 'feature names',
 'filename', 'data module']
>>> iris.data.head(3)
   sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
\Theta5.13.51.40.24.9
                                       3.01.40.2\mathbf{1}4.73.21.30.2\overline{2}>>> iris.target.head(3) # note that the instances are not shuffled
\boldsymbol{\Theta}\Theta\Theta\mathbf{1}\overline{2}\ThetaName: target, dtype: int64
>>> iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

<sup>12</sup> Photos reproduced from the corresponding Wikipedia pages. Iris virginica photo by Frank Mayfield (Creative Commons BY-SA 2.0), Iris versicolor photo by D. Gordon E. Robertson (Creative Commons BY-SA 3.0), Iris setosa photo public domain.

{195}------------------------------------------------

Next we'll split the data and train a logistic regression model on the training set:

```
from sklearn.linear model import LogisticRegression
from sklearn.model_selection import train test split
X = \text{iris.data}[[\text{"petal width (cm)"}]].valuesy = iris.target_names[iris.target] == 'virginica'X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

Let's look at the model's estimated probabilities for flowers with petal widths varying from 0 cm to 3 cm (Figure 4-23): $13$ 

```
X_new = np.linspace(0, 3, 1000).reshape(-1, 1) # reshape to get a column vector
y proba = log_{reg}.predict\_prob(a(x_new))decision_boundary = X_new[y_prob[:, 1] >= 0.5][0, 0]plt.plot(X_new, y_prob[:, 0], "b--", linewidth=2,label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
[...] # beautify the figure: add grid, labels, axis, legend, arrows, and samples
plt.show()
```

![](img/_page_195_Figure_4.jpeg)

Figure 4-23. Estimated probabilities and decision boundary

The petal width of *Iris virginica* flowers (represented as triangles) ranges from 1.4 cm to 2.5 cm, while the other iris flowers (represented by squares) generally have a smaller petal width, ranging from 0.1 cm to 1.8 cm. Notice that there is a bit of overlap. Above about 2 cm the classifier is highly confident that the flower is an Iris virginica (it outputs a high probability for that class), while below 1 cm it is highly

<sup>13</sup> NumPy's reshape() function allows one dimension to be -1, which means "automatic": the value is inferred from the length of the array and the remaining dimensions.

{196}------------------------------------------------

confident that it is not an *Iris virginica* (high probability for the "Not Iris virginica" class). In between these extremes, the classifier is unsure. However, if you ask it to predict the class (using the predict() method rather than the predict\_proba() method), it will return whichever class is the most likely. Therefore, there is a decision boundary at around 1.6 cm where both probabilities are equal to 50%: if the petal width is greater than 1.6 cm the classifier will predict that the flower is an Iris virginica, and otherwise it will predict that it is not (even if it is not very confident):

```
>>> decision boundary
1.6516516516516517
>>> log_{reg.predict}([1.7], [1.5]])array([ True, False])
```

Figure 4-24 shows the same dataset, but this time displaying two features: petal width and length. Once trained, the logistic regression classifier can, based on these two features, estimate the probability that a new flower is an Iris virginica. The dashed line represents the points where the model estimates a 50% probability: this is the model's decision boundary. Note that it is a linear boundary.<sup>14</sup> Each parallel line represents the points where the model outputs a specific probability, from 15% (bottom left) to 90% (top right). All the flowers beyond the top-right line have over 90% chance of being Iris virginica, according to the model.

![](img/_page_196_Figure_3.jpeg)

Figure 4-24. Linear decision boundary

![](img/_page_196_Picture_5.jpeg)

The hyperparameter controlling the regularization strength of a Scikit-Learn LogisticRegression model is not alpha (as in other linear models), but its inverse: C. The higher the value of C, the less the model is regularized.

<sup>14</sup> It is the set of points **x** such that  $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$ , which defines a straight line.

{197}------------------------------------------------

Just like the other linear models, logistic regression models can be regularized using  $\ell_1$  or  $\ell_2$  penalties. Scikit-Learn actually adds an  $\ell_2$  penalty by default.

### **Softmax Regression**

The logistic regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers (as discussed in Chapter 3). This is called softmax regression, or multinomial logistic regression.

The idea is simple: when given an instance x, the softmax regression model first computes a score  $s_k(x)$  for each class k, then estimates the probability of each class by applying the *softmax function* (also called the *normalized exponential*) to the scores. The equation to compute  $s_k(x)$  should look familiar, as it is just like the equation for linear regression prediction (see Equation 4-19).

Equation 4-19. Softmax score for class k

$$
s_k(\mathbf{x}) = \left(\mathbf{\theta}^{(k)}\right)^{\mathsf{T}} \mathbf{x}
$$

Note that each class has its own dedicated parameter vector  $\theta^{(k)}$ . All these vectors are typically stored as rows in a *parameter matrix*  $\Theta$ .

Once you have computed the score of every class for the instance x, you can estimate the probability  $\hat{p}_k$  that the instance belongs to class k by running the scores through the softmax function (Equation 4-20). The function computes the exponential of every score, then normalizes them (dividing by the sum of all the exponentials). The scores are generally called logits or log-odds (although they are actually unnormalized log-odds).

Equation 4-20. Softmax function

$$
\widehat{p}_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp (s_k(\mathbf{x}))}{\sum_{j=1}^K \exp (s_j(\mathbf{x}))}
$$

In this equation:

- $\bullet$  K is the number of classes
- $s(x)$  is a vector containing the scores of each class for the instance  $x$ .
- $\sigma(s(x))_k$  is the estimated probability that the instance x belongs to class k, given the scores of each class for that instance

{198}------------------------------------------------

Just like the logistic regression classifier, by default the softmax regression classifier predicts the class with the highest estimated probability (which is simply the class with the highest score), as shown in Equation 4-21.

Equation 4-21. Softmax regression classifier prediction

 $\hat{y}$  = argmax  $\sigma(\mathbf{s}(\mathbf{x}))_k$  = argmax  $s_k(\mathbf{x})$  = argmax  $((\mathbf{\theta}^{(k)})^T \mathbf{x})$ 

The *argmax* operator returns the value of a variable that maximizes a function. In this equation, it returns the value of k that maximizes the estimated probability  $\sigma(s(x))_k$ .

![](img/_page_198_Picture_4.jpeg)

The softmax regression classifier predicts only one class at a time (i.e., it is multiclass, not multioutput), so it should be used only with mutually exclusive classes, such as different species of plants. You cannot use it to recognize multiple people in one picture.

Now that you know how the model estimates probabilities and makes predictions, let's take a look at training. The objective is to have a model that estimates a high probability for the target class (and consequently a low probability for the other classes). Minimizing the cost function shown in Equation 4-22, called the cross entropy, should lead to this objective because it penalizes the model when it estimates a low probability for a target class. Cross entropy is frequently used to measure how well a set of estimated class probabilities matches the target classes.

Equation 4-22. Cross entropy cost function

 $J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log \left( \hat{p}_k^{(i)} \right)$ 

In this equation,  $y_k^{(i)}$  is the target probability that the  $i^{\text{th}}$  instance belongs to class k. In general, it is either equal to 1 or 0, depending on whether the instance belongs to the class or not.

Notice that when there are just two classes  $(K = 2)$ , this cost function is equivalent to the logistic regression cost function (log loss; see Equation 4-17).

#### **Cross Entropy**

Cross entropy originated from Claude Shannon's information theory. Suppose you want to efficiently transmit information about the weather every day. If there are eight options (sunny, rainy, etc.), you could encode each option using 3 bits, because  $2^3$  = 8. However, if you think it will be sunny almost every day, it would be much more

{199}------------------------------------------------

efficient to code "sunny" on just one bit (0) and the other seven options on four bits (starting with a 1). Cross entropy measures the average number of bits you actually send per option. If your assumption about the weather is perfect, cross entropy will be equal to the entropy of the weather itself (i.e., its intrinsic unpredictability). But if your assumption is wrong (e.g., if it rains often), cross entropy will be greater by an amount called the Kullback-Leibler (KL) divergence.

The cross entropy between two probability distributions p and q is defined as  $H(p,q)$  $= -\sum_{x} p(x) \log q(x)$  (at least when the distributions are discrete). For more details, check out my video on the subject.

The gradient vector of this cost function with regard to  $\theta^{(k)}$  is given by Equation 4-23.

Equation 4-23. Cross entropy gradient vector for class k

$$
\nabla_{\boldsymbol{\theta}}(k) J(\boldsymbol{\Theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \widehat{p}_{k}^{(i)} - y_{k}^{(i)} \right) \mathbf{x}^{(i)}
$$

Now you can compute the gradient vector for every class, then use gradient descent (or any other optimization algorithm) to find the parameter matrix  $\Theta$  that minimizes the cost function.

Let's use softmax regression to classify the iris plants into all three classes. Scikit-Learn's LogisticRegression classifier uses softmax regression automatically when you train it on more than two classes (assuming you use solver="lbfgs", which is the default). It also applies  $\ell_2$  regularization by default, which you can control using the hyperparameter C, as mentioned earlier:

```
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = \text{iris}['target"]X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
softmax_{reg} = LogisticRegression(C=30, random_state=42)softmax reg.fit(X train, y train)
```

So the next time you find an iris with petals that are 5 cm long and 2 cm wide, you can ask your model to tell you what type of iris it is, and it will answer Iris virginica (class 2) with 96% probability (or Iris versicolor with 4% probability):

```
>>> softmax reg.predict([[5, 2]])
array([2])>>> softmax_reg.predict_proba([[5, 2]]).round(2)
array([[0. , 0.04, 0.96]])
```

Figure 4-25 shows the resulting decision boundaries, represented by the background colors. Notice that the decision boundaries between any two classes are linear. The figure also shows the probabilities for the *Iris versicolor* class, represented by the

{200}------------------------------------------------

curved lines (e.g., the line labeled with 0.30 represents the 30% probability boundary). Notice that the model can predict a class that has an estimated probability below 50%. For example, at the point where all decision boundaries meet, all classes have an equal estimated probability of 33%.

![](img/_page_200_Figure_1.jpeg)

Figure 4-25. Softmax regression decision boundaries

In this chapter, you learned various ways to train linear models, both for regression and for classification. You used a closed-form equation to solve linear regression, as well as gradient descent, and you learned how various penalties can be added to the cost function during training to regularize the model. Along the way, you also learned how to plot learning curves and analyze them, and how to implement early stopping. Finally, you learned how logistic regression and softmax regression work. We've opened up the first machine learning black boxes! In the next chapters we will open many more, starting with support vector machines.

### **Exercises**

- 1. Which linear regression training algorithm can you use if you have a training set with millions of features?
- 2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?
- 3. Can gradient descent get stuck in a local minimum when training a logistic regression model?
- 4. Do all gradient descent algorithms lead to the same model, provided you let them run long enough?
- 5. Suppose you use batch gradient descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?

{201}------------------------------------------------

- 6. Is it a good idea to stop mini-batch gradient descent immediately when the validation error goes up?
- 7. Which gradient descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?
- 8. Suppose you are using polynomial regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?
- 9. Suppose you are using ridge regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter  $\alpha$  or reduce it?
- 10. Why would you want to use:
  - a. Ridge regression instead of plain linear regression (i.e., without any regularization)?
  - b. Lasso instead of ridge regression?
  - c. Elastic net instead of lasso regression?
- 11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two logistic regression classifiers or one softmax regression classifier?
- 12. Implement batch gradient descent with early stopping for softmax regression without using Scikit-Learn, only NumPy. Use it on a classification task such as the iris dataset.

Solutions to these exercises are available at the end of this chapter's notebook, at https://homl.info/colab3.

{202}------------------------------------------------
