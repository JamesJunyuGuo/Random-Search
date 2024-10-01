# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch>1.0, click, numpy, torchvision, tqdm, scipy, Pillow, ray``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

If you want to run the MuJoCo experiments, install OpenAI Gym (version 0.9.3) and MuJoCo(version 0.5.7) following the [instructions](https://github.com/openai/gym).

If you want to run the AirFoil experiments, install [XFoil](https://web.mit.edu/drela/Public/web/xfoil/) and make sure the binary is in the `$PATH`.

If you want to run the continous optimization benchmark, install ``Pagmo`` following [esa/pagmo2](https://github.com/esa/pagmo2).

# Usage
Experiment specific parameters are provided as a json file. See the `hc.json` for an example.

To run an example experiment, use the command: 
```bash
python mujoco_experiments.py --param_file=./hc.json
```
