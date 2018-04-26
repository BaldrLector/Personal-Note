# Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate Shift

>**Paper Link**
>[Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

## 1. Introduction

>This part give a view on general gradient descent update rule:
$$\begin{aligned}
\Theta &= \arg \min_\Theta \frac1N \sum_{i=1}^N l (x_i,\Theta)  \\
\end{aligned}$$
$$\frac1m \frac{\partial l(x_i,\Theta)}{\partial \Theta}$$

> Consider two layer condition
$$l = F_2(F_1(u,\Theta_1),\Theta_2)$$
$$ x = F_1(u,\Theta_1)$$
$$l = F_2(x,\Theta_2)$$
$$\Theta_2 \gets \Theta_2 -\frac{\alpha}{m}\sum_{i=1}^m \frac{\partial F_2(x_i,\Theta_2)}{\partial \Theta_2}$$

> This paper also talk about **gradient vanish** problem when using **sigmoid/tanh** as   activate function, this could be solved by replace activate fucntion with **ReLU** and careful **initialization** (even the author never notice, but **He initialization** should work well)
>
>**BN** reduce dependence of gradients on the **scale** of the parameters or of their **initial** values.
>
>Batch Normalization makes it possible to use **saturating nonlinearities** by preventing the network from getting stuck in the saturated modes

> author also defin **internal covariate shift**, which is change distributions of internal nodes of deep network, eliminiating it offer a **faster training**

# 2.Towards Reducing Internal Covariate Shift

>Network training **converges faster** if its inputs are **whitened**
>
>BN will make same whitening for each layer
>
>
>
# 3. Normalization via Mini-Batch Statistics

Paper introduce **Batch Normalizing Transform** as below, where we indicate parameters $\gamma$ and $\beta$ are to learned. **BN depends both training example and other example in mini-batchs.** 

![](./pics/TIM截图20180423174044.png)


During **backpropagation** as below:

![](./pics/TIM截图20180423174225.png)

## 3.1 Training and Inference with Batch-Normalized Networks

- we add $\epsilon$ for smoothing
- we computer $E[x]$ and $Var[x]$ for inference

![](./pics/TIM截图20180423174252.png)

And also, I saw a differnt way to deal this probelm in **CS231N**,which show as below.Here introduce a update way to get two values.

```python

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        pass
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        pass
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    pass

    x, gamma, beta, x_hat, sample_mean, sample_var, eps = cache
    num_train = x.shape[0]

    dx_1 = gamma * dout
    dx_2_b = np.sum((x - sample_mean) * dx_1, axis=0)
    dx_2_a = ((sample_var + eps) ** -0.5) * dx_1
    dx_3_b = (-0.5) * ((sample_var + eps) ** -1.5) * dx_2_b
    dx_4_b = dx_3_b * 1
    dx_5_b = np.ones_like(x) / num_train * dx_4_b
    dx_6_b = 2 * (x - sample_mean) * dx_5_b
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)
    dx_9_b = np.ones_like(x) / num_train * dx_8_b
    dx_10 = dx_9_b + dx_7_a

    dx = dx_10
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta
```



## 3.2 Batch-Normalized Convolutional Networks

## 3.3 Batch Normalization enables higher learning rates

## 3.4 Batch Normalization regularizes the model


