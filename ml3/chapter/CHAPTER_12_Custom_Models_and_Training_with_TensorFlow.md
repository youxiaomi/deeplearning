## **CHAPTER 12 Custom Models and Training** with TensorFlow

Up until now, we've used only TensorFlow's high-level API, Keras, but it already got us pretty far: we built various neural network architectures, including regression and classification nets, Wide & Deep nets, and self-normalizing nets, using all sorts of techniques, such as batch normalization, dropout, and learning rate schedules. In fact, 95% of the use cases you will encounter will not require anything other than Keras (and tf.data; see Chapter 13). But now it's time to dive deeper into TensorFlow and take a look at its lower-level Python API. This will be useful when you need extra control to write custom loss functions, custom metrics, layers, models, initializers, regularizers, weight constraints, and more. You may even need to fully control the training loop itself; for example, to apply special transformations or constraints to the gradients (beyond just clipping them) or to use multiple optimizers for different parts of the network. We will cover all these cases in this chapter, and we will also look at how you can boost your custom models and training algorithms using TensorFlow's automatic graph generation feature. But first, let's take a quick tour of TensorFlow.

### **A Quick Tour of TensorFlow**

As you know, TensorFlow is a powerful library for numerical computation, particularly well suited and fine-tuned for large-scale machine learning (but you can use it for anything else that requires heavy computations). It was developed by the Google Brain team and it powers many of Google's large-scale services, such as Google Cloud Speech, Google Photos, and Google Search. It was open sourced in November 

{431}------------------------------------------------

2015, and it is now the most widely used deep learning library in the industry: countless projects use TensorFlow for all sorts of machine learning tasks, such as image classification, natural language processing, recommender systems, and time series forecasting.

So what does TensorFlow offer? Here's a summary:

- Its core is very similar to NumPy, but with GPU support.
- It supports distributed computing (across multiple devices and servers).
- It includes a kind of just-in-time (JIT) compiler that allows it to optimize computations for speed and memory usage. It works by extracting the *computation* graph from a Python function, optimizing it (e.g., by pruning unused nodes), and running it efficiently (e.g., by automatically running independent operations in parallel).
- Computation graphs can be exported to a portable format, so you can train a TensorFlow model in one environment (e.g., using Python on Linux) and run it in another (e.g., using Java on an Android device).
- It implements reverse-mode autodiff (see Chapter 10 and Appendix B) and provides some excellent optimizers, such as RMSProp and Nadam (see Chapter 11), so you can easily minimize all sorts of loss functions.

TensorFlow offers many more features built on top of these core features: the most important is of course Keras,<sup>2</sup> but it also has data loading and preprocessing ops (tf.data, tf.io, etc.), image processing ops (tf.image), signal processing ops (tf.signal), and more (see Figure 12-1 for an overview of TensorFlow's Python API).

![](img/_page_431_Picture_8.jpeg)

We will cover many of the packages and functions of the Tensor-Flow API, but it's impossible to cover them all, so you should really take some time to browse through the API; you will find that it is quite rich and well documented.

At the lowest level, each TensorFlow operation (op for short) is implemented using highly efficient C++ code.<sup>3</sup> Many operations have multiple implementations called kernels: each kernel is dedicated to a specific device type, such as CPUs, GPUs, or even TPUs *(tensor processing units)*. As you may know, GPUs can dramatically speed up computations by splitting them into many smaller chunks and running them in

<sup>1</sup> However, Facebook's PyTorch library is currently more popular in academia: more papers cite PyTorch than TensorFlow or Keras. Moreover, Google's JAX library is gaining momentum, especially in academia.

<sup>2</sup> TensorFlow includes another deep learning API called the estimators API, but it is now deprecated.

<sup>3</sup> If you ever need to (but you probably won't), you can write your own operations using the C++ API.

{432}------------------------------------------------

parallel across many GPU threads. TPUs are even faster: they are custom ASIC chips built specifically for deep learning operations<sup>4</sup> (we will discuss how to use TensorFlow with GPUs or TPUs in Chapter 19).

![](img/_page_432_Figure_1.jpeg)

Figure 12-1. TensorFlow's Python API

TensorFlow's architecture is shown in Figure 12-2. Most of the time your code will use the high-level APIs (especially Keras and tf.data), but when you need more flexibility you will use the lower-level Python API, handling tensors directly. In any case, TensorFlow's execution engine will take care of running the operations efficiently, even across multiple devices and machines if you tell it to.

TensorFlow runs not only on Windows, Linux, and macOS, but also on mobile devices (using TensorFlow Lite), including both iOS and Android (see Chapter 19). Note that APIs for other languages are also available, if you do not want to use the Python API: there are C++, Java, and Swift APIs. There is even a JavaScript implementation called TensorFlow.js that makes it possible to run your models directly in your browser.

<sup>4</sup> To learn more about TPUs and how they work, check out https://homl.info/tpus.

{433}------------------------------------------------

![](img/_page_433_Figure_0.jpeg)

Figure 12-2. TensorFlow's architecture

There's more to TensorFlow than the library. TensorFlow is at the center of an extensive ecosystem of libraries. First, there's TensorBoard for visualization (see Chapter 10). Next, there's TensorFlow Extended (TFX), which is a set of libraries built by Google to productionize TensorFlow projects: it includes tools for data validation, preprocessing, model analysis, and serving (with TF Serving; see Chapter 19). Google's TensorFlow Hub provides a way to easily download and reuse pretrained neural networks. You can also get many neural network architectures, some of them pretrained, in TensorFlow's model garden. Check out the TensorFlow Resources and https://github.com/jtoy/awesome-tensorflow for more TensorFlow-based projects. You will find hundreds of TensorFlow projects on GitHub, so it is often easy to find existing code for whatever you are trying to do.

![](img/_page_433_Picture_3.jpeg)

More and more ML papers are released along with their implementations, and sometimes even with pretrained models. Check out https://paperswithcode.com to easily find them.

Last but not least, TensorFlow has a dedicated team of passionate and helpful developers, as well as a large community contributing to improving it. To ask technical questions, you should use https://stackoverflow.com and tag your question with tensorflow and python. You can file bugs and feature requests through GitHub. For general discussions, join the TensorFlow Forum.

OK, it's time to start coding!

{434}------------------------------------------------

### **Using TensorFlow like NumPy**

TensorFlow's API revolves around *tensors*, which flow from operation to operation hence the name TensorFlow. A tensor is very similar to a NumPy ndarray: it is usually a multidimensional array, but it can also hold a scalar (a simple value, such as 42). These tensors will be important when we create custom cost functions, custom metrics, custom layers, and more, so let's see how to create and manipulate them.

### **Tensors and Operations**

You can create a tensor with tf.constant(). For example, here is a tensor representing a matrix with two rows and three columns of floats:

```
>>> import tensorflow as tf
>>> t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
\Rightarrow \Rightarrow \pm<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],[4., 5., 6.]], dtype=float32)>
```

Just like an ndarray, a tf. Tensor has a shape and a data type (dtype):

```
>>> t.shape
TensorShape([2, 3])
>>> t.dtype
tf.float32
```

Indexing works much like in NumPy:

```
\gg t[:, 1:]
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([72., 3.][5., 6.]], dtype=float32)>
\Rightarrow t[..., 1, tf.newaxis]
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[2.][5.]], dtype=float32)>
```

Most importantly, all sorts of tensor operations are available:

```
\Rightarrow t + 10<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[11., 12., 13.],[14., 15., 16.], dtype=float32)>
\Rightarrow \frac{1}{2} tf. square(t)
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[1., 4., 9.],[16., 25., 36.], dtype=float32)>
\Rightarrow t \theta tf.transpose(t)
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([14., 32.],[32., 77.]], dtype=float32)>
```

{435}------------------------------------------------

Note that writing  $t + 10$  is equivalent to calling  $tf.add(t, 10)$  (indeed, Python calls the magic method t. \_\_add\_(10), which just calls tf.add(t, 10)). Other operators, like - and \*, are also supported. The @ operator was added in Python 3.5, for matrix multiplication: it is equivalent to calling the tf.matmul() function.

![](img/_page_435_Picture_1.jpeg)

Many functions and classes have aliases. For example, tf.add() and tf.math.add() are the same function. This allows TensorFlow to have concise names for the most common operations<sup>5</sup> while preserving well-organized packages.

A tensor can also hold a scalar value. In this case, the shape is empty:

```
>>> tf.constant(42)
<tf.Tensor: shape=(), dtype=int32, numpy=42>
```

![](img/_page_435_Picture_5.jpeg)

The Keras API has its own low-level API, located in tf.keras.backend. This package is usually imported as K, for conciseness. It used to include functions like K.square(), K.exp(), and K.sqrt(), which you may run across in existing code: this was useful to write portable code back when Keras supported multiple backends, but now that Keras is TensorFlow-only, you should call TensorFlow's low-level API directly (e.g., tf. square() instead of K.square()). Technically K.square() and its friends are still there for backward compatibility, but the documentation of the tf.keras.backend package only lists a handful of utility functions, such as clear\_session() (mentioned in Chapter 10).

You will find all the basic math operations you need  $(tf.add)$ ,  $tf.multiply()$ ,  $tf.\square$ square(),  $tf.\square$ p(),  $tf.\square$ sqrt(), etc.) and most operations that you can find in NumPy (e.g., tf.reshape(), tf.squeeze(), tf.tile()). Some functions have a different name than in NumPy; for instance, tf.reduce\_mean(), tf.reduce\_sum(), tf.reduce\_max(), and tf.math.log() are the equivalent of np.mean(), np.sum(), np.max(), and np.log(). When the name differs, there is usually a good reason for it. For example, in TensorFlow you must write  $tf.transpose(t)$ ; you cannot just write t. T like in NumPy. The reason is that the tf. transpose() function does not do exactly the same thing as NumPy's T attribute: in TensorFlow, a new tensor is created with its own copy of the transposed data, while in NumPy, t.T is just a transposed view on the same data. Similarly, the tf. reduce sum() operation is named this way because its GPU kernel (i.e., GPU implementation) uses a reduce algorithm that does

<sup>5</sup> A notable exception is tf.math.log(), which is commonly used but doesn't have a tf.log() alias, as it might be confused with logging.

{436}------------------------------------------------

not guarantee the order in which the elements are added: because 32-bit floats have limited precision, the result may change ever so slightly every time you call this operation. The same is true of tf. reduce mean() (but of course tf. reduce max() is deterministic).

### **Tensors and NumPy**

Tensors play nice with NumPy: you can create a tensor from a NumPy array, and vice versa. You can even apply TensorFlow operations to NumPy arrays and NumPy operations to tensors:

```
>>> import numpy as np
\Rightarrow a = np.array([2., 4., 5.])
>>> tf.constant(a)
\mathsf{eff}.\mathsf{Tensor}: \mathsf{id}=\mathsf{111}, \mathsf{shape}=\mathsf{(3,)}, \mathsf{dtype}=\mathsf{float64}, \mathsf{numpv}=\mathsf{array}(\mathsf{[2., 4., 5.]}) \times\Rightarrow t.numpy() # or np.array(t)
array([[1., 2., 3.],[4., 5., 6.]], dtype=float32)
\Rightarrow tf.square(a)
<tf.Tensor: id=116, shape=(3,), dtype=float64, numpy=array([4., 16., 25.])>
\Rightarrow np.square(t)
array([[1., 4., 9.],[16., 25., 36.], dtype=float32)
```

![](img/_page_436_Picture_4.jpeg)

Notice that NumPy uses 64-bit precision by default, while Tensor-Flow uses 32-bit. This is because 32-bit precision is generally more than enough for neural networks, plus it runs faster and uses less RAM. So when you create a tensor from a NumPy array, make sure to set dtype=tf.float32.

### **Type Conversions**

Type conversions can significantly hurt performance, and they can easily go unnoticed when they are done automatically. To avoid this, TensorFlow does not perform any type conversions automatically: it just raises an exception if you try to execute an operation on tensors with incompatible types. For example, you cannot add a float tensor and an integer tensor, and you cannot even add a 32-bit float and a 64-bit float:

```
\Rightarrow tf.constant(2.) + tf.constant(40)
[...] InvalidArgumentError: [...] expected to be a float tensor [...]
>>> tf. constant(2.) + tf. constant(40., dtype=tf.float64)[...] InvalidArgumentError: [...] expected to be a float tensor [...]
```

This may be a bit annoying at first, but remember that it's for a good cause! And of course you can use tf.cast() when you really need to convert types:

```
\Rightarrow t2 = tf.constant(40., dtype=tf.float64)
>>> tf. constant(2.0) + tf.cast(t2, tf.float32)<tf.Tensor: id=136, shape=(), dtype=float32, numpy=42.0>
```

{437}------------------------------------------------

### **Variables**

The tf. Tensor values we've seen so far are immutable: we cannot modify them. This means that we cannot use regular tensors to implement weights in a neural network, since they need to be tweaked by backpropagation. Plus, other parameters may also need to change over time (e.g., a momentum optimizer keeps track of past gradients). What we need is a tf. Variable:

```
>>> v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])>>v<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],[4., 5., 6.]], dtype=float32)>
```

A tf. Variable acts much like a tf. Tensor: you can perform the same operations with it, it plays nicely with NumPy as well, and it is just as picky with types. But it can also be modified in place using the assign() method (or assign\_add() or assign sub(), which increment or decrement the variable by the given value). You can also modify individual cells (or slices), by using the cell's (or slice's) assign() method or by using the scatter\_update() or scatter\_nd\_update() methods:

```
v \cdot \text{assignment}(2 * v)# v now equals [[2., 4., 6.], [8., 10., 12.]]
v[0, 1].assign(42)# v now equals [[2., 42., 6.], [8., 10., 12.]]
v[:, 2]. assign([0., 1.]) # v now equals [[2., 42., 0.], [8., 10., 1.]]v.scatter_nd_update(
                       # v now equals [[100., 42., 0.], [8., 10., 200.]]
    indices=[[0, 0], [1, 2]], updates=[100., 200.]
```

Direct assignment will not work:

```
\Rightarrow \lor [1] = [7, 8, 9.][...] TypeError: 'ResourceVariable' object does not support item assignment
```

![](img/_page_437_Picture_7.jpeg)

In practice you will rarely have to create variables manually; Keras provides an add weight() method that will take care of it for you, as you'll see. Moreover, model parameters will generally be updated directly by the optimizers, so you will rarely need to update variables manually.

#### **Other Data Structures**

TensorFlow supports several other data structures, including the following (see the "Other Data Structures" section in this chapter's notebook or Appendix C for more details):

{438}------------------------------------------------

Sparse tensors (tf. SparseTensor)

Efficiently represent tensors containing mostly zeros. The tf. sparse package contains operations for sparse tensors.

Tensor arrays (tf. TensorArray)

Are lists of tensors. They have a fixed length by default but can optionally be made extensible. All tensors they contain must have the same shape and data type.

Ragged tensors (tf.RaggedTensor)

Represent lists of tensors, all of the same rank and data type, but with varying sizes. The dimensions along which the tensor sizes vary are called the *ragged* dimensions. The tf. ragged package contains operations for ragged tensors.

String tensors

Are regular tensors of type tf.string. These represent byte strings, not Unicode strings, so if you create a string tensor using a Unicode string (e.g., a regular Python 3 string like "café"), then it will get encoded to UTF-8 automatically (e.g., b"caf\xc3\xa9"). Alternatively, you can represent Unicode strings using tensors of type tf. int32, where each item represents a Unicode code point (e.g., [99, 97, 102, 233]). The tf. strings package (with an s) contains ops for byte strings and Unicode strings (and to convert one into the other). It's important to note that a tf. string is atomic, meaning that its length does not appear in the tensor's shape. Once you convert it to a Unicode tensor (i.e., a tensor of type tf. int32 holding Unicode code points), the length appears in the shape.

**Sets** 

Are represented as regular tensors (or sparse tensors). For example, tf.constant( $[[1, 2], [3, 4]]$ ) represents the two sets  $\{1, 2\}$  and  $\{3, 4\}$ . More generally, each set is represented by a vector in the tensor's last axis. You can manipulate sets using operations from the tf. sets package.

#### **Oueues**

Store tensors across multiple steps. TensorFlow offers various kinds of queues: basic first-in, first-out (FIFO) queues (FIFOQueue), plus queues that can prioritize some items (PriorityQueue), shuffle their items (RandomShuffleQueue), and batch items of different shapes by padding (PaddingFIFOQueue). These classes are all in the tf.queue package.

With tensors, operations, variables, and various data structures at your disposal, you are now ready to customize your models and training algorithms!

{439}------------------------------------------------

### **Customizing Models and Training Algorithms**

You'll start by creating a custom loss function, which is a straightforward and common use case.

### **Custom Loss Functions**

Suppose you want to train a regression model, but your training set is a bit noisy. Of course, you start by trying to clean up your dataset by removing or fixing the outliers, but that turns out to be insufficient; the dataset is still noisy. Which loss function should you use? The mean squared error might penalize large errors too much and cause your model to be imprecise. The mean absolute error would not penalize outliers as much, but training might take a while to converge, and the trained model might not be very precise. This is probably a good time to use the Huber loss (introduced in Chapter 10) instead of the good old MSE. The Huber loss is available in Keras (just use an instance of the tf.keras.losses.Huber class), but let's pretend it's not there. To implement it, just create a function that takes the labels and the model's predictions as arguments, and uses TensorFlow operations to compute a tensor containing all the losses (one per sample):

```
def huber fn(y true, y pred):
    error = y true - y predis\_small\_error = tf.abs(error) < 1squared loss = tf.\squaresquare(error) / 2
    linear_loss = tf-abs(error) - 0.5return tf.where(is_small_error, squared_loss, linear_loss)
```

![](img/_page_439_Picture_5.jpeg)

For better performance, you should use a vectorized implementation, as in this example. Moreover, if you want to benefit from TensorFlow's graph optimization features, you should use only TensorFlow operations.

It is also possible to return the mean loss instead of the individual sample losses, but this is not recommended as it makes it impossible to use class weights or sample weights when you need them (see Chapter 10).

Now you can use this Huber loss function when you compile the Keras model, then train your model as usual:

```
model.compile(loss=huber fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
```

And that's it! For each batch during training, Keras will call the huber\_fn() function to compute the loss, then it will use reverse-mode autodiff to compute the gradients of the loss with regard to all the model parameters, and finally it will perform a gradient descent step (in this example using a Nadam optimizer). Moreover, it will

{440}------------------------------------------------

keep track of the total loss since the beginning of the epoch, and it will display the mean loss

But what happens to this custom loss when you save the model?

### **Saving and Loading Models That Contain Custom Components**

Saving a model containing a custom loss function works fine, but when you load it, you'll need to provide a dictionary that maps the function name to the actual function. More generally, when you load a model containing custom objects, you need to map the names to the objects:

```
model = tf.keras.models.load model("my model with a custom loss",
                                   custom objects={"huber fn": huber fn})
```

![](img/_page_440_Picture_5.jpeg)

If you decorate the huber\_fn() function with @keras.utils. register keras\_serializable(), it will automatically be available to the load model() function: there's no need to include it in the custom objects dictionary.

With the current implementation, any error between  $-1$  and 1 is considered "small". But what if you want a different threshold? One solution is to create a function that creates a configured loss function:

```
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_{true} - y_{pred}is small error = tf</del>) < thresholdsquared loss = tf.\squaresquare(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2return tf.where(is_small_error, squared_loss, linear_loss)
    return huber fn
```

```
model.compile(loss=create huber(2.0), optimizer="nadam")
```

Unfortunately, when you save the model, the threshold will not be saved. This means that you will have to specify the threshold value when loading the model (note that the name to use is "huber\_fn", which is the name of the function you gave Keras, not the name of the function that created it):

```
model = tf_keras_models.load model()"my model with a custom loss threshold 2",
    custom_objects={"huber_fn": create_huber(2.0)}
\lambda
```

You can solve this by creating a subclass of the tf.keras.losses.Loss class, and then implementing its get config() method:

{441}------------------------------------------------

```
class HuberLoss(tf.keras.losses.Loss):
   def init (self, threshold=1.0, **kwargs):
       self.threshold = threshold
       super(). init (**kwarqs)def call(self, y_true, y_pred):
       error = y_{true} - y_{pred}is\_small\_error = tf.abs(error) < self.thresholdsquared loss = tf.square(error) / 2
       linear loss = self.threshold * tf.abs(error) - self.threshold**2 / 2return tf.where(is_small_error, squared_loss, linear_loss)
   def get_config(self):
       base config = super() . get confiq()return {**base config, "threshold": self.threshold}
```

Let's walk through this code:

- The constructor accepts \*\*kwargs and passes them to the parent constructor, which handles standard hyperparameters: the name of the loss and the reduction algorithm to use to aggregate the individual instance losses. By default this is "AUTO", which is equivalent to "SUM OVER BATCH SIZE": the loss will be the sum of the instance losses, weighted by the sample weights, if any, and divided by the batch size (not by the sum of weights, so this is *not* the weighted mean).<sup>6</sup> Other possible values are "SUM" and "NONE".
- The call () method takes the labels and predictions, computes all the instance losses, and returns them.
- The get config() method returns a dictionary mapping each hyperparameter name to its value. It first calls the parent class's get\_config() method, then adds the new hyperparameters to this dictionary.<sup>7</sup>

You can then use any instance of this class when you compile the model:

model.compile(loss=HuberLoss(2.), optimizer="nadam")

When you save the model, the threshold will be saved along with it; and when you load the model, you just need to map the class name to the class itself:

```
model = tf.keras.models.load model("my model with a custom loss class",
                                   custom_objects={"HuberLoss": HuberLoss})
```

<sup>6</sup> It would not be a good idea to use a weighted mean: if you did, then two instances with the same weight but in different batches would have a different impact on training, depending on the total weight of each batch.

<sup>7</sup> The {\*\*x, [...]} syntax was added in Python 3.5, to merge all the key/value pairs from dictionary x into another dictionary. Since Python 3.9, you can use the nicer  $x \mid y$  syntax instead (where x and y are two dictionaries).

{442}------------------------------------------------

When you save a model, Keras calls the loss instance's get config() method and saves the config in the SavedModel format. When you load the model, it calls the from config() class method on the HuberLoss class: this method is implemented by the base class (Loss) and creates an instance of the class, passing \*\* config to the constructor.

That's it for losses! As you'll see now, custom activation functions, initializers, regularizers, and constraints are not much different.

#### **Custom Activation Functions, Initializers, Regularizers,** and Constraints

Most Keras functionalities, such as losses, regularizers, constraints, initializers, metrics, activation functions, layers, and even full models, can be customized in much the same way. Most of the time, you will just need to write a simple function with the appropriate inputs and outputs. Here are examples of a custom activation function (equivalent to tf.keras.activations.softplus() or tf.nn.softplus()), a custom Glorot initializer (equivalent to tf.keras.initializers.glorot\_normal()), a custom  $\ell_1$  regularizer (equivalent to tf.keras.regularizers.11(0.01)), and a custom constraint that ensures weights are all positive (equivalent to tf.keras. constraints.nonneg() or tf.nn.relu()):

```
def my softplus(z):
    return tf.\text{math.log}(1.0 + tf.\exp(z))def my glorot initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))return tf.random.normal(shape, stddev=stddev, dtype=dtype)
def my_l1_regularizer(weights):
    return tf.reduce sum(tf.abs(0.01 * weights))
def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```

As you can see, the arguments depend on the type of custom function. These custom functions can then be used normally, as shown here:

```
layer = tf.keras. layers. Dense(1, activation=my softplus,kernel_initializer=my_glorot_initializer,
                              kernel_regularizer=my_l1_regularizer,
                              kernel constraint=my positive weights)
```

The activation function will be applied to the output of this Dense layer, and its result will be passed on to the next layer. The layer's weights will be initialized using the value returned by the initializer. At each training step the weights will be passed to the regularization function to compute the regularization loss, which will be added to the main loss to get the final loss used for training. Finally, the constraint function 

{443}------------------------------------------------

will be called after each training step, and the layer's weights will be replaced by the constrained weights.

If a function has hyperparameters that need to be saved along with the model, then you will want to subclass the appropriate class, such as tf. keras. regular izers.Regularizer, tf.keras.constraints.Constraint, tf.keras.initializers. Initializer, or tf.keras.layers.Layer (for any layer, including activation functions). Much as you did for the custom loss, here is a simple class for  $\ell_1$  regularization that saves its factor hyperparameter (this time you do not need to call the parent constructor or the get  $\text{config}()$  method, as they are not defined by the parent class):

```
class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    def _init_(self, factor):
       self.factor = factordef __call_(self, weights):
       return tf.reduce_sum(tf.abs(self.factor * weights))
    def get config(self):
       return {"factor": self.factor}
```

Note that you must implement the call() method for losses, layers (including activation functions), and models, or the call () method for regularizers, initializers, and constraints. For metrics, things are a bit different, as you will see now.

### **Custom Metrics**

Losses and metrics are conceptually not the same thing: losses (e.g., cross entropy) are used by gradient descent to *train* a model, so they must be differentiable (at least at the points where they are evaluated), and their gradients should not be zero everywhere. Plus, it's OK if they are not easily interpretable by humans. In contrast, metrics (e.g., accuracy) are used to *evaluate* a model: they must be more easily interpretable, and they can be nondifferentiable or have zero gradients everywhere.

That said, in most cases, defining a custom metric function is exactly the same as defining a custom loss function. In fact, we could even use the Huber loss function we created earlier as a metric;<sup>8</sup> it would work just fine (and persistence would also work the same way, in this case only saving the name of the function, "huber\_fn", not the threshold):

```
model.compile(loss="mse", optimizer="nadam", metrics=[create huber(2.0)])
```

<sup>8</sup> However, the Huber loss is seldom used as a metric—the MAE or MSE is generally preferred.

{444}------------------------------------------------

For each batch during training, Keras will compute this metric and keep track of its mean since the beginning of the epoch. Most of the time, this is exactly what you want. But not always! Consider a binary classifier's precision, for example. As you saw in Chapter 3, precision is the number of true positives divided by the number of positive predictions (including both true positives and false positives). Suppose the model made five positive predictions in the first batch, four of which were correct: that's 80% precision. Then suppose the model made three positive predictions in the second batch, but they were all incorrect: that's 0% precision for the second batch. If you just compute the mean of these two precisions, you get 40%. But wait a second—that's not the model's precision over these two batches! Indeed, there were a total of four true positives  $(4 + 0)$  out of eight positive predictions  $(5 + 3)$ , so the overall precision is 50%, not 40%. What we need is an object that can keep track of the number of true positives and the number of false positives and that can compute the precision based on these numbers when requested. This is precisely what the tf.keras.metrics.Precision class does:

```
\gg precision = tf.keras.metrics.Precision()
>>> precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
<tf.Tensor: shape=(), dtype=float32, numpy=0.8>
>>> precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
<tf.Tensor: shape=(), dtype=float32, numpy=0.5>
```

In this example, we created a Precision object, then we used it like a function, passing it the labels and predictions for the first batch, then for the second batch (you can optionally pass sample weights as well, if you want). We used the same number of true and false positives as in the example we just discussed. After the first batch, it returns a precision of 80%; then after the second batch, it returns 50% (which is the overall precision so far, not the second batch's precision). This is called a *streaming* metric (or stateful metric), as it is gradually updated, batch after batch.

At any point, we can call the result() method to get the current value of the metric. We can also look at its variables (tracking the number of true and false positives) by using the variables attribute, and we can reset these variables using the reset\_states() method:

```
>>> precision.result()
<tf.Tensor: shape=(), dtype=float32, numpy=0.5>
>>> precision.variables
[<tf.Variable 'true positives:0' [...], numpy=array([4.], dtype=float32)>,
<tf.Variable 'false_positives:0' [...], numpy=array([4.], dtype=float32)>]
>>> precision.reset_states() # both variables get reset to 0.0
```

{445}------------------------------------------------

If you need to define your own custom streaming metric, create a subclass of the tf.keras.metrics.Metric class. Here is a basic example that keeps track of the total Huber loss and the number of instances seen so far. When asked for the result, it returns the ratio, which is just the mean Huber loss:

```
class HuberMetric(tf.keras.metrics.Metric):
   def __init__(self, threshold=1.0, **kwargs):
       super().__init_(**kwargs) # handles base args (e.g., dtype)
       self.threshold = threshold
       self.huber fn = create huber(threshold)
       self.total = self.add_weight("total", initializer="zeros")
       self.count = self.add_weight("count", initializer="zeros")
   def update_state(self, y_true, y_pred, sample_weight=None):
       sample metrics = self.huber fn(y true, y pred)self.total.assign_add(tf.reduce_sum(sample_metrics))
       self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
   def result(self):
       return self.total / self.count
   def get config(self):
       base\_config = super().get\_config()return {**base_config, "threshold": self.threshold}
```

Let's walk through this code:<sup>9</sup>

- The constructor uses the add weight() method to create the variables needed to keep track of the metric's state over multiple batches—in this case, the sum of all Huber losses (total) and the number of instances seen so far (count). You could just create variables manually if you preferred. Keras tracks any tf. Variable that is set as an attribute (and more generally, any "trackable" object, such as layers or models).
- The update\_state() method is called when you use an instance of this class as a function (as we did with the Precision object). It updates the variables, given the labels and predictions for one batch (and sample weights, but in this case we ignore them).
- The result() method computes and returns the final result, in this case the mean Huber metric over all instances. When you use the metric as a function, the update\_state() method gets called first, then the result() method is called, and its output is returned.

<sup>9</sup> This class is for illustration purposes only. A simpler and better implementation would just subclass the tf.keras.metrics.Mean class; see the "Streaming Metrics" section of this chapter's notebook for an example.

{446}------------------------------------------------

- We also implement the get config() method to ensure the threshold gets saved along with the model.
- The default implementation of the reset\_states() method resets all variables to 0.0 (but you can override it if needed).

![](img/_page_446_Picture_2.jpeg)

Keras will take care of variable persistence seamlessly; no action is required.

When you define a metric using a simple function, Keras automatically calls it for each batch, and it keeps track of the mean during each epoch, just like we did manually. So the only benefit of our HuberMetric class is that the threshold will be saved. But of course, some metrics, like precision, cannot simply be averaged over batches: in those cases, there's no other option than to implement a streaming metric.

Now that you've built a streaming metric, building a custom layer will seem like a walk in the park!

### **Custom Layers**

You may occasionally want to build an architecture that contains an exotic layer for which TensorFlow does not provide a default implementation. Or you may simply want to build a very repetitive architecture, in which a particular block of layers is repeated many times, and it would be convenient to treat each block as a single layer. For such cases, you'll want to build a custom layer.

There are some layers that have no weights, such as tf.keras.layers.Flatten or tf.keras.layers.ReLU. If you want to create a custom layer without any weights, the simplest option is to write a function and wrap it in a tf.keras.layers.Lambda layer. For example, the following layer will apply the exponential function to its inputs:

```
exponential layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))
```

This custom layer can then be used like any other layer, using the sequential API, the functional API, or the subclassing API. You can also use it as an activation function, or you could use activation=tf.exp. The exponential layer is sometimes used in the output layer of a regression model when the values to predict have very different scales (e.g., 0.001, 10., 1,000.). In fact, the exponential function is one of the standard activation functions in Keras, so you can just use activation="exponential".

{447}------------------------------------------------

As you might guess, to build a custom stateful layer (i.e., a layer with weights), you need to create a subclass of the tf.keras.layers.Layer class. For example, the following class implements a simplified version of the Dense layer:

```
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(). init (** kwargs)
        self. units = unitsself. activation = tf. keras. activations.get(activation)def build(self, batch input shape):
        self. kernel = self.add_weight(name="kernel", shape=[batch_input_shape[-1], self.units],
           initializer="glorot normal")
        self.bias = self.add-weight(name="bias", shape=[self.units], initializer="zeros")
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
   def get_config(self):
        base\_config = super().get\_config()return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}
```

Let's walk through this code:

- The constructor takes all the hyperparameters as arguments (in this example, units and activation), and importantly it also takes a \*\* kwargs argument. It calls the parent constructor, passing it the kwargs: this takes care of standard arguments such as input\_shape, trainable, and name. Then it saves the hyperparameters as attributes, converting the activation argument to the appropriate activation function using the tf.keras.activations.get() function (it accepts functions, standard strings like "relu" or "swish", or simply None).
- The build() method's role is to create the layer's variables by calling the add weight() method for each weight. The build() method is called the first time the layer is used. At that point, Keras will know the shape of this layer's inputs, and it will pass it to the build() method,<sup>10</sup> which is often necessary to create some of the weights. For example, we need to know the number of neurons in the previous layer in order to create the connection weights matrix (i.e., the "kernel"): this corresponds to the size of the last dimension of the inputs. At the end of the build() method (and only at the end), you must call

<sup>10</sup> The Keras API calls this argument input shape, but since it also includes the batch dimension, I prefer to call it batch input shape.

{448}------------------------------------------------

the parent's build() method: this tells Keras that the layer is built (it just sets  $self.buit = True$ ).

- The call() method performs the desired operations. In this case, we compute the matrix multiplication of the inputs X and the layer's kernel, we add the bias vector, and we apply the activation function to the result, and this gives us the output of the layer.
- The get\_config() method is just like in the previous custom classes. Note that we save the activation function's full configuration by calling tf.keras. activations.serialize().

You can now use a MyDense layer just like any other layer!

![](img/_page_448_Picture_4.jpeg)

Keras automatically infers the output shape, except when the layer is dynamic (as you will see shortly). In this (rare) case, you need to implement the compute\_output\_shape() method, which must return a TensorShape object.

To create a layer with multiple inputs (e.g., Concatenate), the argument to the  $call()$ method should be a tuple containing all the inputs. To create a layer with multiple outputs, the call() method should return the list of outputs. For example, the following toy layer takes two inputs and returns three outputs:

```
class MyMultiLayer(tf.keras.layers.Layer):
   def call(self, X):
       X1. X2 = Xreturn X1 + X2, X1 * X2, X1 / X2
```

This layer may now be used like any other layer, but of course only using the functional and subclassing APIs, not the sequential API (which only accepts layers with one input and one output).

If your layer needs to have a different behavior during training and during testing (e.g., if it uses Dropout or BatchNormalization layers), then you must add a training argument to the call() method and use this argument to decide what to do. For example, let's create a layer that adds Gaussian noise during training (for regularization) but does nothing during testing (Keras has a layer that does the same thing, tf.keras.layers.GaussianNoise):

{449}------------------------------------------------

```
class MyGaussianNoise(tf.keras.layers.Layer):
    def init (self, stddev, **kwargs):
       super().__init__(**kwargs)
       self.stddev = stddev
    def call(self, X, training=False):
       if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noiseelse:
            return X
```

With that, you can now build any custom layer you need! Now let's look at how to create custom models.

#### **Custom Models**

We already looked at creating custom model classes in Chapter 10, when we discussed the subclassing API.<sup>11</sup> It's straightforward: subclass the tf.keras. Model class, create layers and variables in the constructor, and implement the call() method to do whatever you want the model to do. For example, suppose we want to build the model represented in Figure 12-3.

![](img/_page_449_Figure_4.jpeg)

Figure 12-3. Custom model example: an arbitrary model with a custom ResidualBlock layer containing a skip connection

The inputs go through a first dense layer, then through a *residual block* composed of two dense layers and an addition operation (as you will see in Chapter 14, a residual

<sup>11</sup> The name "subclassing API" in Keras usually refers only to the creation of custom models by subclassing, although many other things can be created by subclassing, as you've seen in this chapter.

{450}------------------------------------------------

block adds its inputs to its outputs), then through this same residual block three more times, then through a second residual block, and the final result goes through a dense output layer. Don't worry if this model does not make much sense; it's just an example to illustrate the fact that you can easily build any kind of model you want, even one that contains loops and skip connections. To implement this model, it is best to first create a ResidualBlock layer, since we are going to create a couple of identical blocks (and we might want to reuse it in another model):

```
class ResidualBlock(tf.keras.layers.Layer):
    def _init_(self, n_layers, n_neurons, **kwargs):
        super()._init_(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu",
                                              kernel initializer="he normal")
                       for in range(n layers)]
    def call(self, inputs):
       Z = \text{inputs}for layer in self.hidden:
            Z = \text{layer}(Z)return inputs + Z
```

This layer is a bit special since it contains other layers. This is handled transparently by Keras: it automatically detects that the hidden attribute contains trackable objects (layers in this case), so their variables are automatically added to this layer's list of variables. The rest of this class is self-explanatory. Next, let's use the subclassing API to define the model itself:

```
class ResidualRegressor(tf.keras.Model):
    def _init_(self, output_dim, **kwargs):
        super()._init_(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
                                             kernel initializer="he normal")
        self.block1 = ResidualBlock(2, 30)self.block2 = ResidualBlock(2, 30)self.out = tf.keras.layers.Dense(output dim)
    def call(self, inputs):
       Z = self.hidden1(inputs)for \frac{1}{2} in range(1 + 3):
           Z = self.block1(Z)Z = self.block2(Z)return self.out(Z)
```

We create the layers in the constructor and use them in the call() method. This model can then be used like any other model (compile it, fit it, evaluate it, and use it to make predictions). If you also want to be able to save the model using the save() method and load it using the tf.keras.models.load\_model() function, you must implement the get config() method (as we did earlier) in both the ResidualBlock 

{451}------------------------------------------------

class and the ResidualRegressor class. Alternatively, you can save and load the weights using the save weights() and load weights() methods.

The Model class is a subclass of the Layer class, so models can be defined and used exactly like layers. But a model has some extra functionalities, including of course its compile(), fit(), evaluate(), and predict() methods (and a few variants), plus the get\_layer() method (which can return any of the model's layers by name or by index) and the save() method (and support for tf.keras.models.load\_model() and tf.keras.models.clone\_model()).

![](img/_page_451_Picture_2.jpeg)

If models provide more functionality than layers, why not just define every layer as a model? Well, technically you could, but it is usually cleaner to distinguish the internal components of your model (i.e., layers or reusable blocks of layers) from the model itself (i.e., the object you will train). The former should subclass the Laver class, while the latter should subclass the Model class.

With that, you can naturally and concisely build almost any model that you find in a paper, using the sequential API, the functional API, the subclassing API, or even a mix of these. "Almost" any model? Yes, there are still a few things that we need to look at: first, how to define losses or metrics based on model internals, and second, how to build a custom training loop.

### **Losses and Metrics Based on Model Internals**

The custom losses and metrics we defined earlier were all based on the labels and the predictions (and optionally sample weights). There will be times when you want to define losses based on other parts of your model, such as the weights or activations of its hidden layers. This may be useful for regularization purposes or to monitor some internal aspect of your model.

To define a custom loss based on model internals, compute it based on any part of the model you want, then pass the result to the add\_loss() method. For example, let's build a custom regression MLP model composed of a stack of five hidden layers plus an output layer. This custom model will also have an auxiliary output on top of the upper hidden layer. The loss associated with this auxiliary output will be called the *reconstruction loss* (see Chapter 17): it is the mean squared difference between the reconstruction and the inputs. By adding this reconstruction loss to the main loss, we will encourage the model to preserve as much information as possible through the hidden layers—even information that is not directly useful for the regression task itself. In practice, this loss sometimes improves generalization (it is a regularization loss). It is also possible to add a custom metric using the model's add\_metric()

{452}------------------------------------------------

method. Here is the code for this custom model with a custom reconstruction loss and a corresponding metric:

```
class ReconstructingRegressor(tf.keras.Model):
    def _init_(self, output_dim, **kwargs):
        super(). __init (**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                              kernel_initializer="he_normal")
                       for \lfloor in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction mean = tf.keras.metrics.Mean(
            name="reconstruction error")
    def build(self, batch_input_shape):
        n inputs = batch input shape[-1]
        self. reconstruction = tf. keras. layers. Dense(n inputs)def call(self, inputs, training=False):
        Z = \text{inputs}for layer in self.hidden:
            Z = \text{layer}(Z)reconstruction = self.reconstruct(Z)recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)if training:
            result = self.reconstruction mean(recon loss)
            self.add metric(result)
        return self.out(Z)
```

Let's go through this code:

- The constructor creates the DNN with five dense hidden layers and one dense output layer. We also create a Mean streaming metric to keep track of the reconstruction error during training.
- The build() method creates an extra dense layer that will be used to reconstruct the inputs of the model. It must be created here because its number of units must be equal to the number of inputs, and this number is unknown before the build() method is called.<sup>12</sup>
- The call() method processes the inputs through all five hidden layers, then passes the result through the reconstruction layer, which produces the reconstruction.
- Then the call() method computes the reconstruction loss (the mean squared difference between the reconstruction and the inputs), and adds it to the model's

<sup>12</sup> Due to TensorFlow issue #46858, the call to super().build() may fail in this case, unless the issue was fixed by the time you read this. If not, you need to replace this line with self. built = True.

{453}------------------------------------------------

list of losses using the add loss() method.<sup>13</sup> Notice that we scale down the reconstruction loss by multiplying it by 0.05 (this is a hyperparameter you can tune). This ensures that the reconstruction loss does not dominate the main loss.

- Next, during training only, the call () method updates the reconstruction metric and adds it to the model so it can be displayed. This code example can actually be simplified by calling self.add\_metric(recon\_loss) instead: Keras will automatically track the mean for you.
- Finally, the call() method passes the output of the hidden layers to the output layer and returns its output.

Both the total loss and the reconstruction loss will go down during training:

```
Epoch 1/5363/363 [========] - 1s 820us/step - loss: 0.7640 - reconstruction_error: 1.2728
Epoch 2/5
363/363 [========] - 0s 809us/step - loss: 0.4584 - reconstruction_error: 0.6340
\lceil \dots \rceil
```

In most cases, everything we have discussed so far will be sufficient to implement whatever model you want to build, even with complex architectures, losses, and metrics. However, for some architectures, such as GANs (see Chapter 17), you will have to customize the training loop itself. Before we get there, we must look at how to compute gradients automatically in TensorFlow.

### **Computing Gradients Using Autodiff**

To understand how to use autodiff (see Chapter 10 and Appendix B) to compute gradients automatically, let's consider a simple toy function:

```
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2
```

If you know calculus, you can analytically find that the partial derivative of this function with regard to  $w1$  is 6 \*  $w1 + 2$  \* w2. You can also find that its partial derivative with regard to  $w2$  is  $2 * w1$ . For example, at the point  $(w1, w2) = (5, 3)$ , these partial derivatives are equal to 36 and 10, respectively, so the gradient vector at this point is (36, 10). But if this were a neural network, the function would be much more complex, typically with tens of thousands of parameters, and finding the partial derivatives analytically by hand would be a virtually impossible task. One solution could be to compute an approximation of each partial derivative by measuring how much the function's output changes when you tweak the corresponding parameter by a tiny amount:

<sup>13</sup> You can also call add loss() on any layer inside the model, as the model recursively gathers losses from all of its layers.

{454}------------------------------------------------

```
>> w1, w2 = 5, 3\Rightarrow \Rightarrow eps = 1e-6
\Rightarrow (f(w1 + eps, w2) - f(w1, w2)) / eps
36.000003007075065
\Rightarrow (f(w1, w2 + eps) - f(w1, w2)) / eps
10.000000003174137
```

Looks about right! This works rather well and is easy to implement, but it is just an approximation, and importantly you need to call f() at least once per parameter (not twice, since we could compute  $f(w1, w2)$  just once). Having to call  $f()$  at least once per parameter makes this approach intractable for large neural networks. So instead, we should use reverse-mode autodiff. TensorFlow makes this pretty simple:

```
w1, w2 = tf.Variable(5.), tf.Variable(3.)with tf.GradientTape() as tape:
    z = f(w1, w2)gradients = tape.gradient(z, [w1, w2])
```

We first define two variables w1 and w2, then we create a tf.GradientTape context that will automatically record every operation that involves a variable, and finally we ask this tape to compute the gradients of the result z with regard to both variables [w1, w2]. Let's take a look at the gradients that TensorFlow computed:

```
>>> gradients
[<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
 <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
```

Perfect! Not only is the result accurate (the precision is only limited by the floatingpoint errors), but the gradient() method only goes through the recorded computations once (in reverse order), no matter how many variables there are, so it is incredibly efficient. It's like magic!

![](img/_page_454_Picture_6.jpeg)

In order to save memory, only put the strict minimum inside the tf.GradientTape() block. Alternatively, pause recording by creating a with tape.stop\_recording() block inside the tf.Gradient Tape() block.

The tape is automatically erased immediately after you call its gradient () method, so you will get an exception if you try to call gradient() twice:

```
with tf.GradientTape() as tape:
    z = f(w1, w2)dz_dw1 = tape.gradient(z, w1) # returns tensor 36.0dz dw2 = tape.gradient(z, w2) # raises a RuntimeError!
```

{455}------------------------------------------------

If you need to call gradient() more than once, you must make the tape persistent and delete it each time you are done with it to free resources:<sup>14</sup>

```
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)dz_dw1 = tape.gradient(z, w1) # returns tensor 36.0
dz dw2 = tape.qradient(z, w2) # returns tensor 10.0, works fine now!del tape
```

By default, the tape will only track operations involving variables, so if you try to compute the gradient of z with regard to anything other than a variable, the result will be None:

```
c1, c2 = tf.constant(5.), tf.constant(3.)with tf.GradientTape() as tape:
    z = f(c1, c2)gradients = tape.gradient(z, [c1, c2]) # returns [None, None]
```

However, you can force the tape to watch any tensors you like, to record every operation that involves them. You can then compute gradients with regard to these tensors, as if they were variables:

```
with tf.GradientTape() as tape:
    \text{tape.watch}(c1)tape.watch(c2)z = f(c1, c2)gradients = tape.gradient(z, [c1, c2]) # returns [tensor 36., tensor 10.]
```

This can be useful in some cases, like if you want to implement a regularization loss that penalizes activations that vary a lot when the inputs vary little: the loss will be based on the gradient of the activations with regard to the inputs. Since the inputs are not variables, you'll need to tell the tape to watch them.

Most of the time a gradient tape is used to compute the gradients of a single value (usually the loss) with regard to a set of values (usually the model parameters). This is where reverse-mode autodiff shines, as it just needs to do one forward pass and one reverse pass to get all the gradients at once. If you try to compute the gradients of a vector, for example a vector containing multiple losses, then TensorFlow will compute the gradients of the vector's sum. So if you ever need to get the individual gradients (e.g., the gradients of each loss with regard to the model parameters), you must call the tape's jacobian() method: it will perform reverse-mode autodiff once for each loss in the vector (all in parallel by default). It is even possible to

<sup>14</sup> If the tape goes out of scope, for example when the function that used it returns, Python's garbage collector will delete it for you.

{456}------------------------------------------------

compute second-order partial derivatives (the Hessians, i.e., the partial derivatives of the partial derivatives), but this is rarely needed in practice (see the "Computing Gradients Using Autodiff" section of this chapter's notebook for an example).

In some cases you may want to stop gradients from backpropagating through some part of your neural network. To do this, you must use the tf.stop\_gradient() function. The function returns its inputs during the forward pass (like  $tf.identity()$ ), but it does not let gradients through during backpropagation (it acts like a constant):

```
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop gradient(2 * w1 * w2)with tf.GradientTape() as tape:
    z = f(w1, w2) # the forward pass is not affected by stop_gradient()
gradients = tape.gradient(z, [w1, w2]) # returns [tensor 30., None]
```

Finally, you may occasionally run into some numerical issues when computing gradients. For example, if you compute the gradients of the square root function at  $x =$  $10^{-50}$ , the result will be infinite. In reality, the slope at that point is not infinite, but it's more than 32-bit floats can handle:

```
\Rightarrow x = tf.Variable(1e-50)>>> with tf.GradientTape() as tape:
\ldots z = tf.sqrt(x)
\ddotsc\Rightarrow tape.gradient(z, [x])
[<tf.Tensor: shape=(), dtype=float32, numpy=inf>]
```

To solve this, it's often a good idea to add a tiny value to  $x$  (such as  $10^{-6}$ ) when computing its square root.

The exponential function is also a frequent source of headaches, as it grows extremely fast. For example, the way my\_softplus() was defined earlier is not numerically stable. If you compute my\_softplus(100.0), you will get infinity rather than the correct result (about 100). But it's possible to rewrite the function to make it numerically stable: the softplus function is defined as  $log(1 + exp(z))$ , which is also equal to  $log(1 + exp(-|z|)) + max(z, 0)$  (see the notebook for the mathematical proof) and the advantage of this second form is that the exponential term cannot explode. So, here's a better implementation of the my\_softplus() function:

```
def my softplus(z):
   return tf.math.log(1 + tf.exp(-tf.abs(z))) + tf.maximum(0., z)
```

In some rare cases, a numerically stable function may still have numerically unstable gradients. In such cases, you will have to tell TensorFlow which equation to use for the gradients, rather than letting it use autodiff. For this, you must use the @tf.custom\_gradient decorator when defining the function, and return both the function's usual result and a function that computes the gradients. For example, let's 

{457}------------------------------------------------

update the my softplus() function to also return a numerically stable gradients function:

```
@tf.custom_gradient
def my_softplus(z):
    def my softplus gradients(grads): # grads = backprop'ed from upper layers
        return grads * (1 - 1 / (1 + tf.exp(z))) # stable grads of softplus
    result = tf.\nmathsf{math.log}(1 + tf.\exp(-tf.\nmathsf{abs}(z))) + tf.\nmathsf{maximum}(0., z)return result, my_softplus_gradients
```

If you know differential calculus (see the tutorial notebook on this topic), you can find that the derivative of  $log(1 + exp(z))$  is  $exp(z) / (1 + exp(z))$ . But this form is not stable: for large values of z, it ends up computing infinity divided by infinity, which returns NaN. However, with a bit of algebraic manipulation, you can show that it's also equal to  $1 - 1 / (1 + \exp(z))$ , which is stable. The my\_softplus\_gradients() function uses this equation to compute the gradients. Note that this function will receive as input the gradients that were backpropagated so far, down to the my\_soft plus() function, and according to the chain rule we must multiply them with this function's gradients.

Now when we compute the gradients of the my\_softplus() function, we get the proper result, even for large input values.

Congratulations! You can now compute the gradients of any function (provided it is differentiable at the point where you compute it), even blocking backpropagation when needed, and write your own gradient functions! This is probably more flexibility than you will ever need, even if you build your own custom training loops. You'll see how to do that next

### **Custom Training Loops**

In some cases, the fit() method may not be flexible enough for what you need to do. For example, the Wide & Deep paper we discussed in Chapter 10 uses two different optimizers: one for the wide path and the other for the deep path. Since the fit() method only uses one optimizer (the one that we specify when compiling the model), implementing this paper requires writing your own custom loop.

You may also like to write custom training loops simply to feel more confident that they do precisely what you intend them to do (perhaps you are unsure about some details of the fit() method). It can sometimes feel safer to make everything explicit. However, remember that writing a custom training loop will make your code longer, more error-prone, and harder to maintain.

{458}------------------------------------------------

![](img/_page_458_Picture_0.jpeg)

Unless you're learning or you really need the extra flexibility, you should prefer using the fit() method rather than implementing your own training loop, especially if you work in a team.

First, let's build a simple model. There's no need to compile it, since we will handle the training loop manually:

```
l2 \text{ reg} = \text{tf}. keras. regularizers. l2(0.05)model = tf.keras.models.Sequential(tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                            kernel_regularizer=l2_reg),
    tf.keras.layers.Dense(1, kernel_regularizer=l2_reg)
\overline{1}
```

Next, let's create a tiny function that will randomly sample a batch of instances from the training set (in Chapter 13 we will discuss the tf.data API, which offers a much better alternative):

```
def random batch(X, y, batch size=32):
    idx = np.random.randnint(len(X), size=batch size)return X[idx], y[idx]
```

Let's also define a function that will display the training status, including the number of steps, the total number of steps, the mean loss since the start of the epoch (we will use the Mean metric to compute it), and other metrics:

```
def print status bar(step, total, loss, metrics=None):
   metrics = " - ".join([f''[m.name]: [m.result():.4f]'for m in [loss] + (metrics or [])])
   end = "" if step < total else "n"print(f"\{step\}/\{total\} - " + metrics, end=end)
```

This code is self-explanatory, unless you are unfamiliar with Python string formatting:  $\{m$ .result():.4f} will format the metric's result as a float with four digits after the decimal point, and using  $\Gamma$  (carriage return) along with end="" ensures that the status bar always gets printed on the same line.

With that, let's get down to business! First, we need to define some hyperparameters and choose the optimizer, the loss function, and the metrics (just the MAE in this example):

```
n epochs = 5batch size = 32n_{\text{steps}} = \text{len}(X_{\text{train}}) // batch_size
optimize = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
mean_loss = tf.keras.metrics.Mean(name="mean_loss")
metrics = [tf.keras.metrics.MeanAbsoluteError()]
```

And now we are ready to build the custom loop!

{459}------------------------------------------------

```
for epoch in range(1, n epochs + 1):
   print("Epoch {}/{}".format(epoch, n_epochs))
   for step in range(1, n_steps + 1):
       X batch, y batch = random batch(X train scaled, y train)
       with tf.GradientTape() as tape:
           y pred = model(X_batch, training=True)
           main_loss = tf.readuce_mean(loss_fn(y_batch, y_pred))loss = tf.add_n([main_loss] + model.\n)gradients = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(gradients, model.trainable_variables))
       mean loss(loss)
       for metric in metrics:
           metric(y batch, y pred)
       print_status_bar(step, n_steps, mean_loss, metrics)
   for metric in [mean_loss] + metrics:
       metric.reset states()
```

There's a lot going on in this code, so let's walk through it:

- We create two nested loops: one for the epochs, the other for the batches within an epoch.
- Then we sample a random batch from the training set.
- Inside the tf. Gradient Tape() block, we make a prediction for one batch, using the model as a function, and we compute the loss: it is equal to the main loss plus the other losses (in this model, there is one regularization loss per layer). Since the mean\_squared\_error() function returns one loss per instance, we compute the mean over the batch using tf.reduce\_mean() (if you wanted to apply different weights to each instance, this is where you would do it). The regularization losses are already reduced to a single scalar each, so we just need to sum them (using tf.add\_n(), which sums multiple tensors of the same shape and data type).
- Next, we ask the tape to compute the gradients of the loss with regard to each trainable variable—not all variables!—and we apply them to the optimizer to perform a gradient descent step.
- Then we update the mean loss and the metrics (over the current epoch), and we display the status bar.
- At the end of each epoch, we reset the states of the mean loss and the metrics.

If you want to apply gradient clipping (see Chapter 11), set the optimizer's clipnorm or clipvalue hyperparameter. If you want to apply any other transformation to the gradients, simply do so before calling the apply\_gradients() method. And if you want to add weight constraints to your model (e.g., by setting kernel\_constraint

{460}------------------------------------------------

or bias constraint when creating a layer), you should update the training loop to apply these constraints just after apply gradients(), like so:

```
for variable in model.variables:
    if variable.constraint is not None:
        variable.assign(variable.constraint(variable))
```

![](img/_page_460_Picture_2.jpeg)

Don't forget to set training=True when calling the model in the training loop, especially if your model behaves differently during training and testing (e.g., if it uses BatchNormalization or Dropout). If it's a custom model, make sure to propagate the training argument to the layers that your model calls.

As you can see, there are quite a lot of things you need to get right, and it's easy to make a mistake. But on the bright side, you get full control.

Now that you know how to customize any part of your models<sup>15</sup> and training algorithms, let's see how you can use TensorFlow's automatic graph generation feature: it can speed up your custom code considerably, and it will also make it portable to any platform supported by TensorFlow (see Chapter 19).

### **TensorFlow Functions and Graphs**

Back in TensorFlow 1, graphs were unavoidable (as were the complexities that came with them) because they were a central part of TensorFlow's API. Since TensorFlow 2 (released in 2019), graphs are still there, but not as central, and they're much (much!) simpler to use. To show just how simple, let's start with a trivial function that computes the cube of its input:

```
def cube(x):return x * x 3
```

We can obviously call this function with a Python value, such as an int or a float, or we can call it with a tensor:

```
\Rightarrow \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq \leq 
\Rightarrow cube(tf.constant(2.0))
<tf.Tensor: shape=(), dtype=float32, numpy=8.0>
```

Now, let's use tf. function() to convert this Python function to a TensorFlow function:

<sup>15</sup> With the exception of optimizers, as very few people ever customize these; see the "Custom Optimizers" section in the notebook for an example.

{461}------------------------------------------------

```
>>> tf_cube = tf.function(cube)
>>> tf cube
<tensorflow.python.eager.def_function.Function at 0x7fbfe0c54d50>
```

This TF function can then be used exactly like the original Python function, and it will return the same result (but always as tensors):

```
\Rightarrow tf cube(2)
<tf.Tensor: shape=(), dtype=int32, numpy=8>
>>> tf_cube(tf.constant(2.0))
<tf.Tensor: shape=(), dtype=float32, numpy=8.0>
```

Under the hood, tf. function() analyzed the computations performed by the cube() function and generated an equivalent computation graph! As you can see, it was rather painless (we will look at how this works shortly). Alternatively, we could have used tf. function as a decorator; this is actually more common:

```
@tf.function
def tf_cube(x):return x ** 3
```

The original Python function is still available via the TF function's python\_function attribute, in case you ever need it:

```
>>> tf cube.python function(2)
R
```

TensorFlow optimizes the computation graph, pruning unused nodes, simplifying expressions (e.g.,  $1 + 2$  would get replaced with 3), and more. Once the optimized graph is ready, the TF function efficiently executes the operations in the graph, in the appropriate order (and in parallel when it can). As a result, a TF function will usually run much faster than the original Python function, especially if it performs complex computations.<sup>16</sup> Most of the time you will not really need to know more than that: when you want to boost a Python function, just transform it into a TF function. That's all!

Moreover, if you set jit\_compile=True when calling tf.function(), then Tensor-Flow will use accelerated linear algebra (XLA) to compile dedicated kernels for your graph, often fusing multiple operations. For example, if your TF function calls tf.reduce\_sum(a  $*$  b + c), then without XLA the function would first need to compute a \* b and store the result in a temporary variable, then add c to that variable, and lastly call  $tf$ . reduce\_sum() on the result. With XLA, the whole computation gets compiled into a single kernel, which will compute tf. reduce\_sum(a \* b + c) in one shot, without using any large temporary variable. Not only will this be much faster, it will also use dramatically less RAM.

<sup>16</sup> However, in this trivial example, the computation graph is so small that there is nothing at all to optimize, so tf cube() actually runs much slower than cube().

{462}------------------------------------------------

When you write a custom loss function, a custom metric, a custom layer, or any other custom function and you use it in a Keras model (as we've done throughout this chapter), Keras automatically converts your function into a TF function-no need to use tf. function(). So most of the time, the magic is 100% transparent. And if you want Keras to use XLA, you just need to set jit\_compile=True when calling the compile() method. Easy!

![](img/_page_462_Picture_1.jpeg)

You can tell Keras not to convert your Python functions to TF functions by setting dynamic=True when creating a custom layer or a custom model. Alternatively, you can set run eagerly=True when calling the model's compile() method.

By default, a TF function generates a new graph for every unique set of input shapes and data types and caches it for subsequent calls. For example, if you call tf\_cube(tf.constant(10)), a graph will be generated for int32 tensors of shape []. Then if you call tf\_cube(tf.constant(20)), the same graph will be reused. But if you then call tf\_cube(tf.constant([10, 20])), a new graph will be generated for int32 tensors of shape [2]. This is how TF functions handle polymorphism (i.e., varying argument types and shapes). However, this is only true for tensor arguments: if you pass numerical Python values to a TF function, a new graph will be generated for every distinct value: for example, calling tf\_cube(10) and tf\_cube(20) will generate two graphs.

![](img/_page_462_Picture_4.jpeg)

If you call a TF function many times with different numerical Python values, then many graphs will be generated, slowing down your program and using up a lot of RAM (you must delete the TF function to release it). Python values should be reserved for arguments that will have few unique values, such as hyperparameters like the number of neurons per layer. This allows TensorFlow to better optimize each variant of your model.

### **AutoGraph and Tracing**

So how does TensorFlow generate graphs? It starts by analyzing the Python function's source code to capture all the control flow statements, such as for loops, while loops, and if statements, as well as break, continue, and return statements. This first step is called AutoGraph. The reason TensorFlow has to analyze the source code is that Python does not provide any other way to capture control flow statements: it offers magic methods like \_add\_() and \_mul\_() to capture operators like + and \*, but there are no \_\_while\_() or \_\_if\_() magic methods. After analyzing the function's code, AutoGraph outputs an upgraded version of that function in which all the control flow statements are replaced by the appropriate TensorFlow operations,

{463}------------------------------------------------

such as tf.while loop() for loops and tf.cond() for if statements. For example, in Figure 12-4, AutoGraph analyzes the source code of the sum\_squares() Python function, and it generates the tf\_sum\_squares() function. In this function, the for loop is replaced by the definition of the loop body() function (containing the body of the original for loop), followed by a call to the for\_stmt() function. This call will build the appropriate tf.while\_loop() operation in the computation graph.

![](img/_page_463_Figure_1.jpeg)

Figure 12-4. How TensorFlow generates graphs using AutoGraph and tracing

Next, TensorFlow calls this "upgraded" function, but instead of passing the argument, it passes a *symbolic tensor*—a tensor without any actual value, only a name, a data type, and a shape. For example, if you call sum\_squares(tf.constant(10)), then the tf sum squares() function will be called with a symbolic tensor of type int32 and shape []. The function will run in *graph mode*, meaning that each TensorFlow operation will add a node in the graph to represent itself and its output tensor(s) (as opposed to the regular mode, called *eager execution*, or *eager mode*). In graph mode, TF operations do not perform any computations. Graph mode was the default mode in TensorFlow 1. In Figure 12-4, you can see the tf\_sum\_squares() function being called with a symbolic tensor as its argument (in this case, an int32 tensor of shape []) and the final graph being generated during tracing. The nodes represent operations,

{464}------------------------------------------------

and the arrows represent tensors (both the generated function and the graph are simplified).

![](img/_page_464_Picture_1.jpeg)

In order to view the generated function's source code, you can call tf.autograph.to\_code(sum\_squares.python\_function). The code is not meant to be pretty, but it can sometimes help for debugging.

### **TF Function Rules**

Most of the time, converting a Python function that performs TensorFlow operations into a TF function is trivial: decorate it with @tf.function or let Keras take care of it for you. However, there are a few rules to respect:

- If you call any external library, including NumPy or even the standard library, this call will run only during tracing; it will not be part of the graph. Indeed, a TensorFlow graph can only include TensorFlow constructs (tensors, operations, variables, datasets, and so on). So, make sure you use tf.reduce\_sum() instead of np.sum(), tf.sort() instead of the built-in sorted() function, and so on (unless you really want the code to run only during tracing). This has a few additional implications:
  - If you define a TF function  $f(x)$  that just returns np. random. rand(), a random number will only be generated when the function is traced, so  $f(tf.constant(2.))$  and  $f(tf.constant(3.))$  will return the same random number, but f(tf.constant([2., 3.])) will return a different one. If you replace  $np.random.random()$  with  $tf.randomUniform([])$ , then a new random number will be generated upon every call, since the operation will be part of the graph.
  - If your non-TensorFlow code has side effects (such as logging something or updating a Python counter), then you should not expect those side effects to occur every time you call the TF function, as they will only occur when the function is traced.
  - You can wrap arbitrary Python code in a tf.py\_function() operation, but doing so will hinder performance, as TensorFlow will not be able to do any graph optimization on this code. It will also reduce portability, as the graph will only run on platforms where Python is available (and where the right libraries are installed).
- You can call other Python functions or TF functions, but they should follow the same rules, as TensorFlow will capture their operations in the computation graph. Note that these other functions do not need to be decorated with @tf.function.

{465}------------------------------------------------

- If the function creates a TensorFlow variable (or any other stateful TensorFlow object, such as a dataset or a queue), it must do so upon the very first call, and only then, or else you will get an exception. It is usually preferable to create variables outside of the TF function (e.g., in the build() method of a custom layer). If you want to assign a new value to the variable, make sure you call its assign() method instead of using the = operator.
- The source code of your Python function should be available to TensorFlow. If the source code is unavailable (for example, if you define your function in the Python shell, which does not give access to the source code, or if you deploy only the compiled \*.pyc Python files to production), then the graph generation process will fail or have limited functionality.
- TensorFlow will only capture for loops that iterate over a tensor or a tf.data.Dataset (see Chapter 13). Therefore, make sure you use for i in  $tf.random(x)$  rather than for i in range(x), or else the loop will not be captured in the graph. Instead, it will run during tracing. (This may be what you want if the for loop is meant to build the graph; for example, to create each layer in a neural network.)
- As always, for performance reasons, you should prefer a vectorized implementation whenever you can, rather than using loops.

It's time to sum up! In this chapter we started with a brief overview of TensorFlow, then we looked at TensorFlow's low-level API, including tensors, operations, variables, and special data structures. We then used these tools to customize almost every component in the Keras API. Finally, we looked at how TF functions can boost performance, how graphs are generated using AutoGraph and tracing, and what rules to follow when you write TF functions (if you would like to open the black box a bit further and explore the generated graphs, you will find technical details in Appendix D).

In the next chapter, we will look at how to efficiently load and preprocess data with TensorFlow.

### **Exercises**

- 1. How would you describe TensorFlow in a short sentence? What are its main features? Can you name other popular deep learning libraries?
- 2. Is TensorFlow a drop-in replacement for NumPy? What are the main differences between the two?
- 3. Do you get the same result with  $tf.random(10)$  and  $tfن$ . constant(np.  $area(e(10))$ ?

{466}------------------------------------------------

- 4. Can you name six other data structures available in TensorFlow, beyond regular tensors?
- 5. You can define a custom loss function by writing a function or by subclassing the tf.keras.losses.Loss class. When would you use each option?
- 6. Similarly, you can define a custom metric in a function or as a subclass of tf.keras.metrics.Metric. When would you use each option?
- 7. When should you create a custom layer versus a custom model?
- 8. What are some use cases that require writing your own custom training loop?
- 9. Can custom Keras components contain arbitrary Python code, or must they be convertible to TF functions?
- 10. What are the main rules to respect if you want a function to be convertible to a TF function?
- 11. When would you need to create a dynamic Keras model? How do you do that? Why not make all your models dynamic?
- 12. Implement a custom layer that performs *layer normalization* (we will use this type of layer in Chapter 15):
  - a. The build() method should define two trainable weights  $\alpha$  and  $\beta$ , both of shape input\_shape[-1:] and data type  $tf.f$ loat32.  $\alpha$  should be initialized with 1s, and  $\beta$  with 0s.
  - b. The call() method should compute the mean  $\mu$  and standard deviation  $\sigma$ of each instance's features. For this, you can use tf.nn.moments(inputs, axes = -1, keepdims=True), which returns the mean  $\mu$  and the variance  $\sigma^2$ of all instances (compute the square root of the variance to get the standard deviation). Then the function should compute and return  $\alpha \otimes (X - \mu)/(\sigma + \varepsilon)$  $+$   $\beta$ , where  $\otimes$  represents itemwise multiplication (\*) and  $\varepsilon$  is a smoothing term (a small constant to avoid division by zero, e.g., 0.001).
  - c. Ensure that your custom layer produces the same (or very nearly the same) output as the tf. keras. layers. LayerNormalization layer.
- 13. Train a model using a custom training loop to tackle the Fashion MNIST dataset (see Chapter 10):
  - a. Display the epoch, iteration, mean training loss, and mean accuracy over each epoch (updated at each iteration), as well as the validation loss and accuracy at the end of each epoch.
  - b. Try using a different optimizer with a different learning rate for the upper layers and the lower layers.

Solutions to these exercises are available at the end of this chapter's notebook, at https://homl.info/colab3.

{467}------------------------------------------------

{468}------------------------------------------------
