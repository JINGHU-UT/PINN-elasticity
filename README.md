# Physics-Informed-Neural-Networks for linear elasticity
### Jing Hu (UT Austin)

In this project, the linear elasticity problem is solved with Physics Informed Neural Networks (PINN). The governing equations for isotropic linear elasticity are

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;\sigma_{ij,j}&plus;f_j=0\\&space;\sigma_{ij}=\lambda\sigma_{ij}\epsilon_{kk}&plus;2\mu\epsilon_{ij}\\&space;\epsilon_{ij}=\frac{1}{2}(u_{i,j}&plus;u_{i,j})&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;\sigma_{ij,j}&plus;f_j=0\\&space;\sigma_{ij}=\lambda\sigma_{ij}\epsilon_{kk}&plus;2\mu\epsilon_{ij}\\&space;\epsilon_{ij}=\frac{1}{2}(u_{i,j}&plus;u_{i,j})&space;\end{aligned}" title="\tiny \begin{aligned} \sigma_{ij,j}+f_j=0\\ \sigma_{ij}=\lambda\sigma_{ij}\epsilon_{kk}+2\mu\epsilon_{ij}\\ \epsilon_{ij}=\frac{1}{2}(u_{i,j}+u_{i,j}) \end{aligned}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\sigma_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\sigma_{ij}" title="\tiny \sigma_{ij}" /></a> is the stress tensor, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\epsilon_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\epsilon_{ij}" title="\tiny \epsilon_{ij}" /></a> is the strain tensor, u is the displacement vector, f is the body force vector, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\lambda" title="\tiny \lambda" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\mu" title="\tiny \mu" /></a> are the Lamé parameters and Einstein summation applies. Now we consider the linear elasticity problem on a square domain <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;[0,1]\times[0,1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;[0,1]\times[0,1]" title="\tiny [0,1]\times[0,1]" /></a>. We consider the following boundary conditions:

Top wall

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=\mu\pi(\frac{Q}{4}\cos(\pi&space;x)&space;-\cos(2\pi&space;x))\\&space;t_y&=\lambda&space;Q&space;\sin(\pi&space;x)&space;&plus;2\mu&space;Q&space;\sin(\pi&space;x)&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=\mu\pi(\frac{Q}{4}\cos(\pi&space;x)&space;-\cos(2\pi&space;x))\\&space;t_y&=\lambda&space;Q&space;\sin(\pi&space;x)&space;&plus;2\mu&space;Q&space;\sin(\pi&space;x)&space;\end{aligned}" title="\tiny \begin{aligned} t_x&=\mu\pi(\frac{Q}{4}\cos(\pi x) -\cos(2\pi x))\\ t_y&=\lambda Q \sin(\pi x) +2\mu Q \sin(\pi x) \end{aligned}" /></a>

Right wall

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=0\\&space;t_y&=\mu\pi(-\frac{Q}{4}y^4&plus;\cos(\pi&space;y))&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=0\\&space;t_y&=\mu\pi(-\frac{Q}{4}y^4&plus;\cos(\pi&space;y))&space;\end{aligned}" title="\tiny \begin{aligned} t_x&=0\\ t_y&=\mu\pi(-\frac{Q}{4}y^4+\cos(\pi y)) \end{aligned}" /></a>

Left wall

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=0\\&space;t_y&=-\mu\pi(\frac{Q}{4}y^4&plus;\cos(\pi&space;y))&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;t_x&=0\\&space;t_y&=-\mu\pi(\frac{Q}{4}y^4&plus;\cos(\pi&space;y))&space;\end{aligned}" title="\tiny \begin{aligned} t_x&=0\\ t_y&=-\mu\pi(\frac{Q}{4}y^4+\cos(\pi y)) \end{aligned}" /></a>

Bottom wall

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;u_x&space;=&space;0\\u_y&space;=&space;0&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;u_x&space;=&space;0\\u_y&space;=&space;0&space;\end{aligned}" title="\tiny \begin{aligned} u_x = 0\\u_y = 0 \end{aligned}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;t_\bullet" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;t_\bullet" title="\tiny t_\bullet" /></a> is the traction prescribed on the boundaries and the Q is the load magnitude. We consider the following body force:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;f_{x}&space;&=\lambda\left[4&space;\pi^{2}&space;\cos&space;(2&space;\pi&space;x)&space;\sin&space;(\pi&space;y)-\pi&space;\cos&space;(\pi&space;x)&space;Q&space;y^{3}\right]&space;\\&space;&&plus;\mu\left[9&space;\pi^{2}&space;\cos&space;(2&space;\pi&space;x)&space;\sin&space;(\pi&space;y)-\pi&space;\cos&space;(\pi&space;x)&space;Q&space;y^{3}\right]\\&space;f_{y}&space;&=\lambda\left[-3&space;\sin&space;(\pi&space;x)&space;Q&space;y^{2}&plus;2&space;\pi^{2}&space;\sin&space;(2&space;\pi&space;x)&space;\cos&space;(\pi&space;y)\right]&space;\\&space;&&plus;\mu\left[-6&space;\sin&space;(\pi&space;x)&space;Q&space;y^{2}&plus;2&space;\pi^{2}&space;\sin&space;(2&space;\pi&space;x)&space;\cos&space;(\pi&space;y)&plus;\pi^{2}&space;\sin&space;(\pi&space;x)&space;Q&space;y^{4}&space;/&space;4\right]&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;f_{x}&space;&=\lambda\left[4&space;\pi^{2}&space;\cos&space;(2&space;\pi&space;x)&space;\sin&space;(\pi&space;y)-\pi&space;\cos&space;(\pi&space;x)&space;Q&space;y^{3}\right]&space;\\&space;&&plus;\mu\left[9&space;\pi^{2}&space;\cos&space;(2&space;\pi&space;x)&space;\sin&space;(\pi&space;y)-\pi&space;\cos&space;(\pi&space;x)&space;Q&space;y^{3}\right]\\&space;f_{y}&space;&=\lambda\left[-3&space;\sin&space;(\pi&space;x)&space;Q&space;y^{2}&plus;2&space;\pi^{2}&space;\sin&space;(2&space;\pi&space;x)&space;\cos&space;(\pi&space;y)\right]&space;\\&space;&&plus;\mu\left[-6&space;\sin&space;(\pi&space;x)&space;Q&space;y^{2}&plus;2&space;\pi^{2}&space;\sin&space;(2&space;\pi&space;x)&space;\cos&space;(\pi&space;y)&plus;\pi^{2}&space;\sin&space;(\pi&space;x)&space;Q&space;y^{4}&space;/&space;4\right]&space;\end{aligned}" title="\tiny \begin{aligned} f_{x} &=\lambda\left[4 \pi^{2} \cos (2 \pi x) \sin (\pi y)-\pi \cos (\pi x) Q y^{3}\right] \\ &+\mu\left[9 \pi^{2} \cos (2 \pi x) \sin (\pi y)-\pi \cos (\pi x) Q y^{3}\right]\\ f_{y} &=\lambda\left[-3 \sin (\pi x) Q y^{2}+2 \pi^{2} \sin (2 \pi x) \cos (\pi y)\right] \\ &+\mu\left[-6 \sin (\pi x) Q y^{2}+2 \pi^{2} \sin (2 \pi x) \cos (\pi y)+\pi^{2} \sin (\pi x) Q y^{4} / 4\right] \end{aligned}" /></a>

In the numerical implementation, we output both displacements and stresses and adopt the following loss function to prescribe boundary conditions and enforce the governing equation.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\begin{aligned}&space;\mathcal{L}&space;&=\left|u-u^{*}\right|_{Dirichlet&space;BC}&plus;\left|t-t^{*}\right|_{Neumann&space;BC}&space;\\&space;&&plus;\left|\sigma_{ij,j}&plus;f_j^*\right|_{collocation&space;points}&space;\\&space;&&plus;\left|\lambda\sigma_{ij}\epsilon_{kk}&plus;2\mu\epsilon_{ij}-\sigma_{ij}\right|_{collocation&space;points}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\begin{aligned}&space;\mathcal{L}&space;&=\left|u-u^{*}\right|_{Dirichlet&space;BC}&plus;\left|t-t^{*}\right|_{Neumann&space;BC}&space;\\&space;&&plus;\left|\sigma_{ij,j}&plus;f_j^*\right|_{collocation&space;points}&space;\\&space;&&plus;\left|\lambda\sigma_{ij}\epsilon_{kk}&plus;2\mu\epsilon_{ij}-\sigma_{ij}\right|_{collocation&space;points}&space;\end{aligned}" title="\tiny \begin{aligned} \mathcal{L} &=\left|u-u^{*}\right|_{Dirichlet BC}+\left|t-t^{*}\right|_{Neumann BC} \\ &+\left|\sigma_{ij,j}+f_j^*\right|_{collocation points} \\ &+\left|\lambda\sigma_{ij}\epsilon_{kk}+2\mu\epsilon_{ij}-\sigma_{ij}\right|_{collocation points} \end{aligned}" /></a>

For model parameters: <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\lambda=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\lambda=1" title="\tiny \lambda=1" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;\mu=0.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;\mu=0.5" title="\tiny \mu=0.5" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\tiny&space;Q=4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\tiny&space;Q=4" title="\tiny Q=4" /></a>, we have the following results

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/u_PINN.png" width="400"/> <img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/u_anal.png" width="400"/>

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/v_PINN.png" width="400"/> <img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/v_anal.png" width="400"/>

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/sxx_PINN.png" width="400"/> <img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/sxx_anal.png" width="400"/>

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/syy_PINN.png" width="400"/> <img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/syy_anal.png" width="400"/>

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/sxy_PINN.png" width="400"/> <img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/sxy_anal.png" width="400"/>

The decrease of the loss function is shown in the following figure.

<img src="https://github.com/JINGHU-UT/PINN-elasticity/blob/main/output/loss.png" width="600"/>

> **Note:** The implementations were developed and tested on colab with TensorFlow 1.9 and hardware accelerator choosen as GPU.