## Part II. Neural Networks and Deep Learning

| 10. Introduction to Artificial Neural Networks with Keras | 299 |
|-----------------------------------------------------------|-----|
| From Biological to Artificial Neurons                     | 300 |
| <b>Biological Neurons</b>                                 | 301 |
| Logical Computations with Neurons                         | 303 |
| The Perceptron                                            | 304 |
| The Multilayer Perceptron and Backpropagation             | 309 |
| Regression MLPs                                           | 313 |
| <b>Classification MLPs</b>                                | 315 |
| Implementing MLPs with Keras                              | 317 |

{9}------------------------------------------------

| Building an Image Classifier Using the Sequential API | 318 |
|-------------------------------------------------------|-----|
| Building a Regression MLP Using the Sequential API    | 328 |
| Building Complex Models Using the Functional API      | 329 |
| Using the Subclassing API to Build Dynamic Models     | 336 |
| Saving and Restoring a Model                          | 337 |
| <b>Using Callbacks</b>                                | 338 |
| Using TensorBoard for Visualization                   | 340 |
| Fine-Tuning Neural Network Hyperparameters            | 344 |
| Number of Hidden Layers                               | 349 |
| Number of Neurons per Hidden Layer                    | 350 |
| Learning Rate, Batch Size, and Other Hyperparameters  | 351 |
| Exercises                                             | 353 |
| 11. Training Deep Neural Networks.                    | 357 |
| The Vanishing/Exploding Gradients Problems            | 358 |
| Glorot and He Initialization                          | 359 |
| <b>Better Activation Functions</b>                    | 361 |
| <b>Batch Normalization</b>                            | 367 |
| <b>Gradient Clipping</b>                              | 372 |
| Reusing Pretrained Layers                             | 373 |
| Transfer Learning with Keras                          | 375 |
| <b>Unsupervised Pretraining</b>                       | 377 |
| Pretraining on an Auxiliary Task                      | 378 |
| <b>Faster Optimizers</b>                              | 379 |
| Momentum                                              | 379 |
| Nesterov Accelerated Gradient                         | 381 |
| AdaGrad                                               | 382 |
| RMSProp                                               | 383 |
| Adam                                                  | 384 |
| AdaMax                                                | 385 |
| Nadam                                                 | 386 |
| AdamW                                                 | 386 |
| Learning Rate Scheduling                              | 388 |
| Avoiding Overfitting Through Regularization           | 392 |
| $\ell_1$ and $\ell_2$ Regularization                  | 393 |
| Dropout                                               | 394 |
| Monte Carlo (MC) Dropout                              | 397 |
| Max-Norm Regularization                               | 399 |
| Summary and Practical Guidelines                      | 400 |
| Exercises                                             | 402 |

{10}------------------------------------------------

| 12. Custom Models and Training with TensorFlow                           |     |
|--------------------------------------------------------------------------|-----|
| A Quick Tour of TensorFlow                                               | 403 |
| Using TensorFlow like NumPy                                              | 407 |
| Tensors and Operations                                                   | 407 |
| Tensors and NumPy                                                        | 409 |
| <b>Type Conversions</b>                                                  | 409 |
| Variables                                                                | 410 |
| Other Data Structures                                                    | 410 |
| Customizing Models and Training Algorithms                               | 412 |
| <b>Custom Loss Functions</b>                                             | 412 |
| Saving and Loading Models That Contain Custom Components                 | 413 |
| Custom Activation Functions, Initializers, Regularizers, and Constraints | 415 |
| <b>Custom Metrics</b>                                                    | 416 |
| Custom Layers                                                            | 419 |
| Custom Models                                                            | 422 |
| Losses and Metrics Based on Model Internals                              | 424 |
| Computing Gradients Using Autodiff                                       | 426 |
| <b>Custom Training Loops</b>                                             | 430 |
| TensorFlow Functions and Graphs                                          | 433 |
| AutoGraph and Tracing                                                    | 435 |
| <b>TF Function Rules</b>                                                 | 437 |
| Exercises                                                                | 438 |
| 13. Loading and Preprocessing Data with TensorFlow                       | 441 |
| The tf.data API                                                          | 442 |
| Chaining Transformations                                                 | 443 |
| Shuffling the Data                                                       | 445 |
| Interleaving Lines from Multiple Files                                   | 446 |
| Preprocessing the Data                                                   | 448 |
| Putting Everything Together                                              | 449 |
| Prefetching                                                              | 450 |
| Using the Dataset with Keras                                             | 452 |
| The TFRecord Format                                                      | 453 |
| Compressed TFRecord Files                                                | 454 |
| A Brief Introduction to Protocol Buffers                                 | 454 |
| <b>TensorFlow Protobufs</b>                                              | 456 |
| Loading and Parsing Examples                                             | 457 |
| Handling Lists of Lists Using the SequenceExample Protobuf               | 459 |
| Keras Preprocessing Layers                                               | 459 |
| The Normalization Layer                                                  | 460 |
| The Discretization Layer                                                 | 463 |

{11}------------------------------------------------

| The CategoryEncoding Layer<br>The StringLookup Layer         | 463<br>465 |
|--------------------------------------------------------------|------------|
| The Hashing Layer                                            | 466        |
| <b>Encoding Categorical Features Using Embeddings</b>        | 466        |
| <b>Text Preprocessing</b>                                    | 471        |
| Using Pretrained Language Model Components                   | 473        |
| <b>Image Preprocessing Layers</b>                            | 474        |
| The TensorFlow Datasets Project                              | 475        |
| Exercises                                                    | 477        |
| 14. Deep Computer Vision Using Convolutional Neural Networks | 479        |
| The Architecture of the Visual Cortex                        | 480        |
| <b>Convolutional Layers</b>                                  | 481        |
| Filters                                                      | 484        |
| <b>Stacking Multiple Feature Maps</b>                        | 485        |
| Implementing Convolutional Layers with Keras                 | 487        |
| Memory Requirements                                          | 490        |
| Pooling Layers                                               | 491        |
| Implementing Pooling Layers with Keras                       | 493        |
| <b>CNN</b> Architectures                                     | 495        |
| LeNet-5                                                      | 498        |
| AlexNet                                                      | 499        |
| GoogLeNet                                                    | 502        |
| VGGNet                                                       | 505        |
| <b>ResNet</b>                                                | 505        |
| Xception                                                     | 509        |
| <b>SENet</b>                                                 | 510        |
| Other Noteworthy Architectures                               | 512        |
| Choosing the Right CNN Architecture                          | 514        |
| Implementing a ResNet-34 CNN Using Keras                     | 515        |
| Using Pretrained Models from Keras                           | 516        |
| Pretrained Models for Transfer Learning                      | 518        |
| Classification and Localization                              | 521        |
| <b>Object Detection</b>                                      | 523        |
| Fully Convolutional Networks                                 | 525        |
| You Only Look Once                                           | 527        |
| <b>Object Tracking</b>                                       | 530        |
| Semantic Segmentation                                        | 531        |
| Exercises                                                    | 535        |

{12}------------------------------------------------

| 15. Processing Sequences Using RNNs and CNNs.                    | 537        |
|------------------------------------------------------------------|------------|
| Recurrent Neurons and Layers                                     | 538        |
| Memory Cells                                                     | 540        |
| <b>Input and Output Sequences</b>                                | 541        |
| <b>Training RNNs</b>                                             | 542        |
| Forecasting a Time Series                                        | 543        |
| The ARMA Model Family                                            | 549        |
| Preparing the Data for Machine Learning Models                   | 552        |
| Forecasting Using a Linear Model                                 | 555        |
| Forecasting Using a Simple RNN                                   | 556        |
| Forecasting Using a Deep RNN                                     | 557        |
| Forecasting Multivariate Time Series                             | 559        |
| Forecasting Several Time Steps Ahead                             | 560        |
| Forecasting Using a Sequence-to-Sequence Model                   | 562        |
| Handling Long Sequences                                          | 565        |
| Fighting the Unstable Gradients Problem                          | 565        |
| Tackling the Short-Term Memory Problem                           | 568        |
| Exercises                                                        | 576        |
| 16. Natural Language Processing with RNNs and Attention.         | 577        |
| Generating Shakespearean Text Using a Character RNN              | 578        |
| Creating the Training Dataset                                    | 579        |
| Building and Training the Char-RNN Model                         | 581        |
| Generating Fake Shakespearean Text                               | 582        |
| <b>Stateful RNN</b>                                              | 584        |
| Sentiment Analysis                                               | 587        |
| Masking                                                          | 590        |
| Reusing Pretrained Embeddings and Language Models                | 593        |
| An Encoder-Decoder Network for Neural Machine Translation        | 595        |
| <b>Bidirectional RNNs</b>                                        | 601        |
| Beam Search                                                      | 603        |
| <b>Attention Mechanisms</b>                                      | 604        |
| Attention Is All You Need: The Original Transformer Architecture | 609        |
| An Avalanche of Transformer Models                               | 620        |
| <b>Vision Transformers</b>                                       | 624        |
| Hugging Face's Transformers Library                              | 629        |
| Exercises                                                        | 633        |
| 17. Autoencoders, GANs, and Diffusion Models.                    |            |
|                                                                  |            |
| <b>Efficient Data Representations</b>                            | 635<br>637 |

{13}------------------------------------------------

| <b>Stacked Autoencoders</b>                                                  | 640 |
|------------------------------------------------------------------------------|-----|
| Implementing a Stacked Autoencoder Using Keras                               | 641 |
| Visualizing the Reconstructions                                              | 642 |
| Visualizing the Fashion MNIST Dataset                                        | 643 |
| Unsupervised Pretraining Using Stacked Autoencoders                          | 644 |
| <b>Tying Weights</b>                                                         | 645 |
| Training One Autoencoder at a Time                                           | 646 |
| Convolutional Autoencoders                                                   | 648 |
| Denoising Autoencoders                                                       | 649 |
| Sparse Autoencoders                                                          | 651 |
| Variational Autoencoders                                                     | 654 |
| <b>Generating Fashion MNIST Images</b>                                       | 658 |
| Generative Adversarial Networks                                              | 659 |
| The Difficulties of Training GANs                                            | 663 |
| Deep Convolutional GANs                                                      | 665 |
| Progressive Growing of GANs                                                  | 668 |
| StyleGANs                                                                    | 671 |
| <b>Diffusion Models</b>                                                      | 673 |
| Exercises                                                                    | 681 |
|                                                                              | 683 |
| 18. Reinforcement Learning<br>Learning to Optimize Rewards                   | 684 |
| Policy Search                                                                | 685 |
| Introduction to OpenAI Gym                                                   | 687 |
| <b>Neural Network Policies</b>                                               | 691 |
|                                                                              | 693 |
| Evaluating Actions: The Credit Assignment Problem<br><b>Policy Gradients</b> | 694 |
| <b>Markov Decision Processes</b>                                             | 699 |
| Temporal Difference Learning                                                 | 703 |
| Q-Learning                                                                   | 704 |
| <b>Exploration Policies</b>                                                  | 706 |
| Approximate Q-Learning and Deep Q-Learning                                   | 707 |
| Implementing Deep Q-Learning                                                 | 708 |
| Deep Q-Learning Variants                                                     | 713 |
| Fixed Q-value Targets                                                        | 713 |
| Double DQN                                                                   | 714 |
| Prioritized Experience Replay                                                | 714 |
| Dueling DQN                                                                  | 715 |
| Overview of Some Popular RL Algorithms                                       | 716 |
|                                                                              |     |
| Exercises                                                                    | 720 |

{14}------------------------------------------------

| 19. Training and Deploying TensorFlow Models at Scale   |     |
|---------------------------------------------------------|-----|
| Serving a TensorFlow Model                              | 722 |
| <b>Using TensorFlow Serving</b>                         | 722 |
| Creating a Prediction Service on Vertex AI              | 732 |
| Running Batch Prediction Jobs on Vertex AI              | 739 |
| Deploying a Model to a Mobile or Embedded Device        | 741 |
| Running a Model in a Web Page                           | 744 |
| Using GPUs to Speed Up Computations                     | 746 |
| Getting Your Own GPU                                    | 747 |
| Managing the GPU RAM                                    | 749 |
| Placing Operations and Variables on Devices             | 752 |
| Parallel Execution Across Multiple Devices              | 753 |
| Training Models Across Multiple Devices                 | 756 |
| Model Parallelism                                       | 756 |
| Data Parallelism                                        | 759 |
| Training at Scale Using the Distribution Strategies API | 765 |
| Training a Model on a TensorFlow Cluster                | 766 |
| Running Large Training Jobs on Vertex AI                | 770 |
| Hyperparameter Tuning on Vertex AI                      | 772 |
| Exercises                                               | 776 |
| Thank You!                                              | 777 |
| A. Machine Learning Project Checklist.                  |     |
| B. Autodiff.                                            |     |
| C. Special Data Structures.                             |     |
| D. TensorFlow Graphs                                    | 801 |
| Index.                                                  |     |

{15}------------------------------------------------

{16}------------------------------------------------
