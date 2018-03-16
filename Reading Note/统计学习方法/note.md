<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.css" integrity="sha384-TEMocfGvRuD1rIAacqrknm5BQZ7W7uWitoih+jMNFXQIbNl16bO8OZmylH/Vi/Ei" crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.9.0/katex.min.js" integrity="sha384-jmxIlussZWB7qCuB+PgKG1uLjjxbVVIayPJwi6cG6Zb4YKq0JIw+OMnkkEC7kYCq" crossorigin="anonymous"></script>

katex.render("c = \\pm\\sqrt{a^2 + b^2}", element);

<span class="katex">\alpha</span>

# Note 1:

## Linear Regression



$\begin{aligned}
H&=X\theta\\
J_\theta &= \frac1{2m} (X\theta-Y)^T(X\theta-Y)\\
\triangledown_\theta J &= \frac{1}{m} X^T (X\theta-Y)\\
gradient\ \ descent:\\
\theta_t &\gets \theta_{t-1} - \alpha \triangledown_\theta J 
\end{aligned}$


## SVM

>线性可分的SVM

- **函数间隔与几何间隔**

$$\hat\gamma_i = y_i (w \cdot x_i + b)$$
$$\hat \gamma  = \min_{i=1,2...N} \hat \gamma_i $$
$$ \gamma= \hat \gamma / ||w||  $$

- **间隔最大化**

$$\begin{aligned}
&max_{w,b}  \gamma\\
&s.t.\ \ y_i (\frac{w}{||w||}\cdot x + \frac{b}{||w||}) \ge \gamma , i=1,2...,N\\
thus, we \ let \ \gamma=1\\
& max_{w,b} \frac12 ||w||^2\\
&s.t. y_i(w\cdot x_i+b)-1 \ge 0
\end{aligned}$$

- **对偶算法**

$$\begin{aligned}
\mathcal L(w,b,\alpha)&=\frac12||w||^2 - \sum_{i=1}^N \alpha_i y_i (w \cdot x+b) +\sum_{i=1}^N\alpha_i\\
\triangledown_w \mathcal L&=w-\sum_{i=1}^N\alpha_iy_i = 0\\
\triangledown_b L &= \sum_{i=1}^N \alpha_i y_i =0\\
thus \\
\mathcal L &= \frac12 \sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i x_j) - \sum_{i=1}^N \alpha_iy_i (\sum_{j=1}^N\alpha_jy_jx_jx_i+b) +\sum_{i=1}^N \alpha_i 
\end{aligned}$$

- **软间隔**

$$\begin{aligned}
min_{w,b,\xi}\  \frac12 ||w||^2 + C\sum_{i=1}^N \xi_i\\
s.t.\ \ y_i (\frac{w}{||w||}\cdot x + \frac{b}{||w||}) \ge 1-\xi_i , i=1,2...,N\\
\\
\end{aligned}$$

重新进行上面的推导，发现$\xi$被消去，并且可以用$\alpha$表示，于是计算过程相同

- **合页损失函数（hinge loss）**

$$\bm L = \sum_{i=1}^N max(0,1-y_i(w\cdot x+b))+ \lambda R_w$$








