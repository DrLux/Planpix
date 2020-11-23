This repo is from my Master's degree thesis work develped at Addfor s.p.a
I used PlaNet to prove that model-based DRL can overcome the model-free algorithms in terms of sample efficiency.
My implementation of PlaNet is based on the [Kaixhin](https://github.com/Kaixhin/PlaNet) one, but I reach better results. I also experiment with a regularizer based on DAE to reduce the gap between the real and the predicted rewards. 
The company asks me to note publish that feature, but you can find all the explanations in my [blog article](https://drlux.github.io/planpix.html)  (you can also contact me).

![Full trained agent](https://raw.githubusercontent.com/DrLux/Planpix/master/images/agent_in_action.gif?token=ADY2SMJQXOZK3BJN3EGTLDK7YJX66)

PlaNet
======
General overview of Planet model architecture. If you want a full explanation, click on it!
[![Blog_article](https://drlux.github.io/masterDegree/Diapositiva16.JPG)](https://drlux.github.io/planpix.html)


Results
------------
![ceetah_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/ceetah_planet_vs_ddpg.jpg?token=ADY2SMLWUAPFPPU4JNXKQS27YJYGS)
![cartpole_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/cartpole_planet_vs_ddpg.jpg?token=ADY2SMPTHHRUXMDSSM6UBR27YJYCQ)
![reacher_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/reacher_planet_vs_ddpg.jpg?token=ADY2SMM4XWCNZGRRW3JZWJC7YJYHU)
![walker_planet_vs_ddpg](https://raw.githubusercontent.com/DrLux/Planpix/master/images/walker_planet_vs_ddpg.jpg?token=ADY2SMNTFGC7XWTOZY7XZTK7YJYJ2)
![my_planet_vs_soa](https://raw.githubusercontent.com/DrLux/Planpix/master/images/soa.png?token=ADY2SMLIPNL4M2F66FJAWAC7YJYLQ)

Comparison's data are form:
[Curl: Contrastive unsupervised representations for reinforcement learning.](https://proceedings.icml.cc/static/paper_files/icml/2020/5951-Paper.pdf) Laskin, M., Srinivas, A., & Abbeel, P. (2020, July)


Requirements
------------

- Python 3
- [DeepMind Control Suite](https://github.com/deepmind/dm_control) (optional)
- [Gym](https://gym.openai.com/)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)


Links
-----

- [Overcoming the limits of DRL using a model-based approach](https://drlux.github.io/planpix.html)
- [Introducing PlaNet: A Deep Planning Network for Reinforcement Learning](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)
- [Kaixhin/planet](https://github.com/Kaixhin/PlaNet)
- [google-research/planet](https://github.com/google-research/planet)
- [Curl: Contrastive unsupervised representations for reinforcement learning](https://proceedings.icml.cc/static/paper_files/icml/2020/5951-Paper.pdf)


Acknowledgements
----------------

- [@Kaixhin](https://github.com/Kaixhin/PlaNet) for its reimplementation of [google-research/planet](https://github.com/google-research/planet) 

References
----------

[1] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  
[2] [Overcoming the limits of DRL using a model-based approach](https://drlux.github.io/planpix.html)

